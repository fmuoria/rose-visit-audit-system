"""
ROSE Women Leaders Visit Audit System
Automated audit system for detecting suspicious patterns in beneficiary visits
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import geodesic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease
import re
import warnings
warnings.filterwarnings('ignore')

class VisitAuditSystem:
    def __init__(self, similarity_threshold=0.7, upload_delay_threshold=7):
        """
        Initialize the audit system with configurable thresholds
        
        Args:
            similarity_threshold (float): Threshold for flagging similar locations/stories (0.7 = 70%)
            upload_delay_threshold (int): Days threshold for flagging delayed uploads
        """
        self.similarity_threshold = similarity_threshold
        self.upload_delay_threshold = upload_delay_threshold
        self.audit_results = {}
    
    def load_data(self, data_source):
        """
        Load visit data from CSV file or pandas DataFrame
        """
        if isinstance(data_source, str):
            self.df = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy()
        else:
            raise ValueError("Data source must be CSV file path or pandas DataFrame")
        
        # Clean and prepare data
        self._prepare_data()
        return self
    
    def _prepare_data(self):
        """
        Clean and prepare data for audit analysis
        """
        # Convert date columns
        self.df['Visitation Date'] = pd.to_datetime(self.df['Visitation Date'])
        self.df['Visitation: Created Date'] = pd.to_datetime(self.df['Visitation: Created Date'])
        
        # Calculate upload delay
        self.df['Upload Delay Days'] = (
            self.df['Visitation: Created Date'] - self.df['Visitation Date']
        ).dt.days
        
        # Clean text fields
        self.df['Story Summary'] = self.df['Story Summary'].fillna('').astype(str)
        self.df['Visitation: Created By'] = self.df['Visitation: Created By'].fillna('Unknown')
        
        # Clean numeric fields
        self.df['Visit Location (Latitude)'] = pd.to_numeric(self.df['Visit Location (Latitude)'], errors='coerce')
        self.df['Visit Location (Longitude)'] = pd.to_numeric(self.df['Visit Location (Longitude)'], errors='coerce')
        
    def audit_location_similarity(self):
        """
        Detect suspiciously similar visit locations by trainer
        """
        location_flags = []
        
        for trainer in self.df['Visitation: Created By'].unique():
            trainer_visits = self.df[self.df['Visitation: Created By'] == trainer].copy()
            
            if len(trainer_visits) < 2:
                continue
                
            # Calculate pairwise distances between all visits
            similar_locations = []
            for i, row1 in trainer_visits.iterrows():
                for j, row2 in trainer_visits.iterrows():
                    if i >= j:  # Avoid duplicate comparisons
                        continue
                    
                    lat1, lon1 = row1['Visit Location (Latitude)'], row1['Visit Location (Longitude)']
                    lat2, lon2 = row2['Visit Location (Latitude)'], row2['Visit Location (Longitude)']
                    
                    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
                        continue
                    
                    # Calculate distance in meters
                    distance = geodesic((lat1, lon1), (lat2, lon2)).meters
                    
                    # Flag if locations are suspiciously close (within 50 meters)
                    if distance < 50:
                        similar_locations.append({
                            'visit_1_index': i,
                            'visit_2_index': j,
                            'distance_meters': distance,
                            'accounts': f"{row1['Account Name']} & {row2['Account Name']}"
                        })
            
            # Calculate similarity percentage for trainer
            total_comparisons = len(trainer_visits) * (len(trainer_visits) - 1) / 2
            similar_count = len(similar_locations)
            similarity_percentage = (similar_count / total_comparisons) if total_comparisons > 0 else 0
            
            if similarity_percentage >= self.similarity_threshold * 0.1:  # Adjusted threshold for location
                location_flags.append({
                    'trainer': trainer,
                    'similarity_percentage': similarity_percentage,
                    'similar_locations': similar_locations,
                    'total_visits': len(trainer_visits),
                    'flagged_pairs': similar_count
                })
        
        return location_flags
    
    def audit_story_similarity(self):
        """
        Detect suspiciously similar story summaries by trainer using TF-IDF and cosine similarity
        """
        story_flags = []
        
        for trainer in self.df['Visitation: Created By'].unique():
            trainer_visits = self.df[self.df['Visitation: Created By'] == trainer].copy()
            stories = trainer_visits['Story Summary'].tolist()
            
            if len(stories) < 2:
                continue
            
            # Remove very short stories (likely incomplete)
            valid_stories = [s for s in stories if len(s.strip()) > 20]
            
            if len(valid_stories) < 2:
                continue
            
            # Calculate TF-IDF similarity matrix
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=1000,
                lowercase=True
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(valid_stories)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Find highly similar story pairs
                similar_stories = []
                for i in range(len(similarity_matrix)):
                    for j in range(i + 1, len(similarity_matrix)):
                        similarity_score = similarity_matrix[i][j]
                        
                        if similarity_score >= self.similarity_threshold:
                            similar_stories.append({
                                'story_1_index': trainer_visits.index[i],
                                'story_2_index': trainer_visits.index[j],
                                'similarity_score': similarity_score,
                                'story_1_preview': valid_stories[i][:100] + '...',
                                'story_2_preview': valid_stories[j][:100] + '...'
                            })
                
                # Calculate overall story authenticity metrics
                avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
                story_lengths = [len(s.split()) for s in valid_stories]
                avg_story_length = np.mean(story_lengths)
                story_diversity = len(set(valid_stories)) / len(valid_stories)
                
                if len(similar_stories) > 0 or avg_similarity > 0.5:
                    story_flags.append({
                        'trainer': trainer,
                        'average_similarity': avg_similarity,
                        'story_diversity': story_diversity,
                        'average_length': avg_story_length,
                        'similar_stories': similar_stories,
                        'total_stories': len(valid_stories),
                        'authenticity_score': self._calculate_story_authenticity(
                            avg_similarity, story_diversity, avg_story_length
                        )
                    })
                    
            except Exception as e:
                print(f"Error processing stories for trainer {trainer}: {e}")
        
        return story_flags
    
    def _calculate_story_authenticity(self, avg_similarity, story_diversity, avg_length):
        """
        Calculate story authenticity score (0-100, higher is more authentic)
        """
        # Penalize high similarity and reward diversity and reasonable length
        similarity_penalty = min(avg_similarity * 100, 50)  # Max 50 points penalty
        diversity_bonus = story_diversity * 30  # Max 30 points bonus
        length_score = min(avg_length / 50 * 20, 20)  # Target ~50 words, max 20 points
        
        authenticity = 100 - similarity_penalty + diversity_bonus + length_score
        return max(0, min(100, authenticity))  # Clamp to 0-100
    
    def audit_upload_delays(self):
        """
        Detect suspicious patterns in upload delays
        """
        delay_flags = []
        
        for trainer in self.df['Visitation: Created By'].unique():
            trainer_visits = self.df[self.df['Visitation: Created By'] == trainer].copy()
            
            # Analyze upload delay patterns
            delays = trainer_visits['Upload Delay Days'].dropna()
            
            if len(delays) == 0:
                continue
            
            # Flag visits with excessive delays
            excessive_delays = trainer_visits[
                trainer_visits['Upload Delay Days'] > self.upload_delay_threshold
            ]
            
            # Calculate delay statistics
            avg_delay = delays.mean()
            max_delay = delays.max()
            delay_consistency = delays.std()  # Lower std = more consistent
            
            if len(excessive_delays) > 0 or avg_delay > self.upload_delay_threshold:
                delay_flags.append({
                    'trainer': trainer,
                    'average_delay_days': avg_delay,
                    'max_delay_days': max_delay,
                    'delay_consistency': delay_consistency,
                    'excessive_delay_count': len(excessive_delays),
                    'total_visits': len(trainer_visits),
                    'excessive_delay_percentage': len(excessive_delays) / len(trainer_visits) * 100
                })
        
        return delay_flags
    
    def audit_income_growth_anomalies(self):
        """
        Detect suspicious patterns in income growth reporting
        """
        income_flags = []
        
        for trainer in self.df['Visitation: Created By'].unique():
            trainer_visits = self.df[self.df['Visitation: Created By'] == trainer].copy()
            
            # Clean income data
            trainer_visits['Current Monthly Income'] = pd.to_numeric(trainer_visits['Current Monthly Income'], errors='coerce')
            trainer_visits['Last Month Income'] = pd.to_numeric(trainer_visits['Last Month Income'], errors='coerce')
            
            # Calculate growth percentages
            valid_income = trainer_visits.dropna(subset=['Current Monthly Income', 'Last Month Income'])
            
            if len(valid_income) < 2:
                continue
            
            # Calculate growth rates
            valid_income['Growth_Rate'] = (
                (valid_income['Current Monthly Income'] - valid_income['Last Month Income']) / 
                valid_income['Last Month Income'].replace(0, np.nan) * 100
            ).fillna(0)
            
            growth_rates = valid_income['Growth_Rate'].dropna()
            
            if len(growth_rates) == 0:
                continue
            
            # Detect anomalies
            suspicious_patterns = []
            
            # 1. Unrealistic growth rates (>500% or <-90%)
            extreme_growth = valid_income[
                (valid_income['Growth_Rate'] > 500) | (valid_income['Growth_Rate'] < -90)
            ]
            
            # 2. Identical growth rates (suspicious uniformity)
            growth_values = growth_rates.round(1).values
            unique_growth_rates = len(set(growth_values))
            total_growth_entries = len(growth_values)
            growth_diversity = unique_growth_rates / total_growth_entries if total_growth_entries > 0 else 1
            
            # 3. Suspiciously round numbers (multiples of 10 or 25)
            round_numbers = valid_income[
                (valid_income['Current Monthly Income'] % 10 == 0) |
                (valid_income['Current Monthly Income'] % 25 == 0)
            ]
            round_number_percentage = len(round_numbers) / len(valid_income)
            
            # 4. Identical income values across different beneficiaries
            income_values = valid_income['Current Monthly Income'].values
            unique_incomes = len(set(income_values))
            income_diversity = unique_incomes / len(income_values) if len(income_values) > 0 else 1
            
            # Flag if suspicious patterns detected
            if (len(extreme_growth) > 0 or 
                growth_diversity < 0.5 or 
                round_number_percentage > 0.7 or
                income_diversity < 0.6):
                
                income_flags.append({
                    'trainer': trainer,
                    'total_income_entries': len(valid_income),
                    'extreme_growth_count': len(extreme_growth),
                    'growth_diversity': growth_diversity,
                    'income_diversity': income_diversity,
                    'round_numbers_percentage': round_number_percentage,
                    'average_growth_rate': growth_rates.mean(),
                    'median_growth_rate': growth_rates.median(),
                    'growth_std': growth_rates.std(),
                    'suspicious_score': self._calculate_income_suspicion_score(
                        len(extreme_growth), growth_diversity, income_diversity, round_number_percentage
                    )
                })
        
        return income_flags
    
    def _calculate_income_suspicion_score(self, extreme_count, growth_diversity, income_diversity, round_percentage):
        """
        Calculate income suspicion score (0-100, higher is more suspicious)
        """
        score = 0
        
        # Extreme growth penalty
        score += min(extreme_count * 15, 40)
        
        # Low diversity penalties
        if growth_diversity < 0.3:
            score += 25
        elif growth_diversity < 0.5:
            score += 15
            
        if income_diversity < 0.4:
            score += 20
        elif income_diversity < 0.6:
            score += 10
        
        # Round numbers penalty
        if round_percentage > 0.8:
            score += 15
        elif round_percentage > 0.6:
            score += 10
        
        return min(score, 100)
    
    def audit_temporal_clustering(self):
        """
        Detect suspicious patterns in visit timing and frequency
        """
        temporal_flags = []
        
        for trainer in self.df['Visitation: Created By'].unique():
            trainer_visits = self.df[self.df['Visitation: Created By'] == trainer].copy()
            trainer_visits = trainer_visits.sort_values('Visitation Date')
            
            if len(trainer_visits) < 5:  # Need minimum visits for pattern analysis
                continue
            
            # Extract temporal features
            trainer_visits['Visit_Hour'] = trainer_visits['Visitation Date'].dt.hour
            trainer_visits['Visit_Day'] = trainer_visits['Visitation Date'].dt.day_name()
            trainer_visits['Visit_Week'] = trainer_visits['Visitation Date'].dt.isocalendar().week
            
            suspicious_patterns = []
            
            # 1. Same time clustering (too many visits at exact same hour)
            hour_counts = trainer_visits['Visit_Hour'].value_counts()
            max_hour_frequency = hour_counts.max()
            hour_clustering_score = max_hour_frequency / len(trainer_visits)
            
            # 2. Day of week patterns (should avoid Mondays, have good distribution)
            day_counts = trainer_visits['Visit_Day'].value_counts()
            monday_visits = day_counts.get('Monday', 0)
            
            # Check if too concentrated on specific days
            max_day_frequency = day_counts.max()
            day_clustering_score = max_day_frequency / len(trainer_visits)
            
            # 3. Batch uploading pattern (many visits uploaded same day)
            upload_dates = trainer_visits['Visitation: Created Date'].dt.date
            upload_counts = upload_dates.value_counts()
            max_upload_batch = upload_counts.max()
            batch_upload_score = max_upload_batch / len(trainer_visits)
            
            # 4. Visit frequency consistency (should be roughly daily except Mondays)
            trainer_visits['Days_Between_Visits'] = trainer_visits['Visitation Date'].diff().dt.days
            avg_days_between = trainer_visits['Days_Between_Visits'].mean()
            visit_consistency = trainer_visits['Days_Between_Visits'].std()
            
            # 5. Time gaps analysis (unusual long gaps)
            long_gaps = trainer_visits[trainer_visits['Days_Between_Visits'] > 7]
            
            # Flag suspicious temporal patterns
            if (hour_clustering_score > 0.4 or  # 40% of visits at same hour
                day_clustering_score > 0.6 or   # 60% of visits on same day
                batch_upload_score > 0.5 or     # 50% uploaded same day
                monday_visits > len(trainer_visits) * 0.2 or  # More than 20% on Mondays
                len(long_gaps) > len(trainer_visits) * 0.3):  # More than 30% with long gaps
                
                temporal_flags.append({
                    'trainer': trainer,
                    'total_visits': len(trainer_visits),
                    'hour_clustering_score': hour_clustering_score,
                    'most_common_hour': hour_counts.index[0],
                    'day_clustering_score': day_clustering_score,
                    'most_common_day': day_counts.index[0],
                    'monday_visits_count': monday_visits,
                    'monday_percentage': (monday_visits / len(trainer_visits)) * 100,
                    'batch_upload_score': batch_upload_score,
                    'largest_upload_batch': max_upload_batch,
                    'average_days_between_visits': avg_days_between,
                    'visit_consistency_std': visit_consistency,
                    'long_gaps_count': len(long_gaps),
                    'temporal_suspicion_score': self._calculate_temporal_suspicion_score(
                        hour_clustering_score, day_clustering_score, batch_upload_score, 
                        monday_visits / len(trainer_visits), len(long_gaps) / len(trainer_visits)
                    )
                })
        
        return temporal_flags
    
    def _calculate_temporal_suspicion_score(self, hour_cluster, day_cluster, batch_upload, monday_ratio, long_gap_ratio):
        """
        Calculate temporal suspicion score (0-100, higher is more suspicious)
        """
        score = 0
        
        # Hour clustering penalty
        if hour_cluster > 0.5:
            score += 25
        elif hour_cluster > 0.3:
            score += 15
        
        # Day clustering penalty
        if day_cluster > 0.7:
            score += 20
        elif day_cluster > 0.5:
            score += 12
        
        # Batch upload penalty
        if batch_upload > 0.6:
            score += 20
        elif batch_upload > 0.4:
            score += 12
        
        # Monday visits penalty (should be minimal)
        if monday_ratio > 0.3:
            score += 15
        elif monday_ratio > 0.2:
            score += 8
        
        # Long gaps penalty
        if long_gap_ratio > 0.4:
            score += 20
        elif long_gap_ratio > 0.2:
            score += 10
        
        return min(score, 100)
    
    def calculate_visit_confidence(self):
        """
        Calculate confidence score for each individual visit
        """
        visit_scores = []
        
        # Get audit results
        location_flags = {f['trainer']: f for f in self.audit_location_similarity()}
        story_flags = {f['trainer']: f for f in self.audit_story_similarity()}
        delay_flags = {f['trainer']: f for f in self.audit_upload_delays()}
        income_flags = {f['trainer']: f for f in self.audit_income_growth_anomalies()}
        temporal_flags = {f['trainer']: f for f in self.audit_temporal_clustering()}
        
        for _, visit in self.df.iterrows():
            trainer = visit['Visitation: Created By']
            confidence_score = 100  # Start with full confidence
            
            # Location confidence penalty
            if trainer in location_flags:
                confidence_score -= location_flags[trainer]['similarity_percentage'] * 30
            
            # Story confidence penalty
            if trainer in story_flags:
                story_authenticity = story_flags[trainer]['authenticity_score']
                confidence_score -= (100 - story_authenticity) * 0.3
            
            # Upload delay penalty
            if visit['Upload Delay Days'] > self.upload_delay_threshold:
                delay_penalty = min((visit['Upload Delay Days'] / self.upload_delay_threshold) * 20, 40)
                confidence_score -= delay_penalty
            
            # Income anomaly penalty
            if trainer in income_flags:
                income_suspicion = income_flags[trainer]['suspicious_score']
                confidence_score -= income_suspicion * 0.2
            
            # Temporal clustering penalty
            if trainer in temporal_flags:
                temporal_suspicion = temporal_flags[trainer]['temporal_suspicion_score']
                confidence_score -= temporal_suspicion * 0.15
            
            # Data completeness bonus/penalty
            required_fields = ['Account Name', 'Story Summary', 'Visit Location (Latitude)', 
                             'Visit Location (Longitude)', 'Current Monthly Income']
            complete_fields = sum(1 for field in required_fields if pd.notna(visit[field]) and visit[field] != '')
            completeness_score = (complete_fields / len(required_fields)) * 10
            confidence_score += completeness_score
            
            visit_scores.append({
                'visit_index': visit.name,
                'trainer': trainer,
                'account_name': visit['Account Name'],
                'visit_date': visit['Visitation Date'],
                'confidence_score': max(0, min(100, confidence_score))  # Clamp to 0-100
            })
        
        return visit_scores
    
    def calculate_trainer_confidence(self):
        """
        Calculate overall confidence score for each trainer
        """
        visit_scores = self.calculate_visit_confidence()
        trainer_scores = {}
        
        for score in visit_scores:
            trainer = score['trainer']
            if trainer not in trainer_scores:
                trainer_scores[trainer] = []
            trainer_scores[trainer].append(score['confidence_score'])
        
        trainer_confidence = []
        for trainer, scores in trainer_scores.items():
            avg_confidence = np.mean(scores)
            min_confidence = np.min(scores)
            low_confidence_count = sum(1 for s in scores if s < 60)
            
            trainer_confidence.append({
                'trainer': trainer,
                'average_confidence': avg_confidence,
                'minimum_confidence': min_confidence,
                'low_confidence_visits': low_confidence_count,
                'total_visits': len(scores),
                'low_confidence_percentage': (low_confidence_count / len(scores)) * 100,
                'audit_priority': self._calculate_audit_priority(avg_confidence, low_confidence_count, len(scores))
            })
        
        return sorted(trainer_confidence, key=lambda x: x['average_confidence'])
    
    def _calculate_audit_priority(self, avg_confidence, low_conf_count, total_visits):
        """
        Calculate audit priority (High, Medium, Low)
        """
        if avg_confidence < 50 or (low_conf_count / total_visits) > 0.5:
            return "HIGH"
        elif avg_confidence < 70 or (low_conf_count / total_visits) > 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_audit_report(self, output_file=None):
        """
        Generate comprehensive audit report
        """
        print("🔍 ROSE Women Leaders Visit Audit Report")
        print("=" * 60)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Visits Analyzed: {len(self.df)}")
        print(f"Total Trainers: {self.df['Visitation: Created By'].nunique()}")
        print()
        
        # Location Analysis
        location_flags = self.audit_location_similarity()
        print("📍 LOCATION SIMILARITY ANALYSIS")
        print("-" * 40)
        if location_flags:
            for flag in location_flags:
                print(f"⚠️  Trainer: {flag['trainer']}")
                print(f"   Similar locations: {flag['similarity_percentage']:.1%}")
                print(f"   Total visits: {flag['total_visits']}")
                print(f"   Flagged pairs: {flag['flagged_pairs']}")
                print()
        else:
            print("✅ No suspicious location patterns detected")
        print()
        
        # Story Analysis
        story_flags = self.audit_story_similarity()
        print("📝 STORY SIMILARITY ANALYSIS")
        print("-" * 40)
        if story_flags:
            for flag in story_flags:
                print(f"⚠️  Trainer: {flag['trainer']}")
                print(f"   Authenticity Score: {flag['authenticity_score']:.1f}/100")
                print(f"   Average Similarity: {flag['average_similarity']:.2f}")
                print(f"   Story Diversity: {flag['story_diversity']:.2f}")
                print(f"   Similar story pairs: {len(flag['similar_stories'])}")
                print()
        else:
            print("✅ No suspicious story patterns detected")
        print()
        
        # Upload Delay Analysis
        delay_flags = self.audit_upload_delays()
        print("⏱️  UPLOAD DELAY ANALYSIS")
        print("-" * 40)
        if delay_flags:
            for flag in delay_flags:
                print(f"⚠️  Trainer: {flag['trainer']}")
                print(f"   Average delay: {flag['average_delay_days']:.1f} days")
                print(f"   Max delay: {flag['max_delay_days']} days")
                print(f"   Excessive delays: {flag['excessive_delay_percentage']:.1f}%")
                print()
        else:
            print("✅ No suspicious upload delay patterns detected")
        print()
        
        # Income Growth Analysis
        income_flags = self.audit_income_growth_anomalies()
        print("💰 INCOME GROWTH ANOMALY ANALYSIS")
        print("-" * 40)
        if income_flags:
            for flag in income_flags:
                print(f"⚠️  Trainer: {flag['trainer']}")
                print(f"   Suspicion Score: {flag['suspicious_score']:.1f}/100")
                print(f"   Growth Diversity: {flag['growth_diversity']:.2f}")
                print(f"   Income Diversity: {flag['income_diversity']:.2f}")
                print(f"   Extreme Growth Count: {flag['extreme_growth_count']}")
                print(f"   Round Numbers: {flag['round_numbers_percentage']:.1%}")
                print()
        else:
            print("✅ No suspicious income growth patterns detected")
        print()
        
        # Temporal Clustering Analysis
        temporal_flags = self.audit_temporal_clustering()
        print("⏰ TEMPORAL CLUSTERING ANALYSIS")
        print("-" * 40)
        if temporal_flags:
            for flag in temporal_flags:
                print(f"⚠️  Trainer: {flag['trainer']}")
                print(f"   Temporal Suspicion: {flag['temporal_suspicion_score']:.1f}/100")
                print(f"   Most Common Hour: {flag['most_common_hour']}:00")
                print(f"   Monday Visits: {flag['monday_percentage']:.1f}%")
                print(f"   Batch Upload Score: {flag['batch_upload_score']:.2f}")
                print(f"   Long Gaps: {flag['long_gaps_count']} visits")
                print()
        else:
            print("✅ No suspicious temporal patterns detected")
        print()
        
        # Trainer Confidence Rankings
        trainer_confidence = self.calculate_trainer_confidence()
        print("🎯 TRAINER AUDIT PRIORITY RANKING")
        print("-" * 40)
        for trainer in trainer_confidence:
            priority = trainer['audit_priority']
            emoji = "🔴" if priority == "HIGH" else "🟡" if priority == "MEDIUM" else "🟢"
            print(f"{emoji} {trainer['trainer']} - {priority} PRIORITY")
            print(f"   Average Confidence: {trainer['average_confidence']:.1f}%")
            print(f"   Low Confidence Visits: {trainer['low_confidence_visits']}/{trainer['total_visits']}")
            print()
        
        # Summary Statistics
        visit_scores = self.calculate_visit_confidence()
        low_confidence_visits = [v for v in visit_scores if v['confidence_score'] < 60]
        
        print("📊 AUDIT SUMMARY")
        print("-" * 40)
        print(f"Low Confidence Visits (< 60%): {len(low_confidence_visits)}")
        print(f"Trainers Requiring Audit: {sum(1 for t in trainer_confidence if t['audit_priority'] in ['HIGH', 'MEDIUM'])}")
        print(f"Average Visit Confidence: {np.mean([v['confidence_score'] for v in visit_scores]):.1f}%")
        
        if output_file:
            # Save detailed results to CSV
            visit_df = pd.DataFrame(visit_scores)
            trainer_df = pd.DataFrame(trainer_confidence)
            
            with pd.ExcelWriter(output_file) as writer:
                visit_df.to_excel(writer, sheet_name='Visit_Scores', index=False)
                trainer_df.to_excel(writer, sheet_name='Trainer_Confidence', index=False)
                self.df.to_excel(writer, sheet_name='Raw_Data', index=False)
            
            print(f"\n💾 Detailed results saved to: {output_file}")

# Example usage
if __name__ == "__main__":
    # Initialize audit system
    auditor = VisitAuditSystem(
        similarity_threshold=0.7,  # 70% similarity threshold
        upload_delay_threshold=7   # 7 days upload delay threshold
    )
    
    # Example of how to use with your data
    # auditor.load_data('visit_data.csv')
    # auditor.generate_audit_report('audit_results.xlsx')
    
    print("ROSE Visit Audit System Ready!")
    print("Usage:")
    print("1. auditor.load_data('your_data.csv')")
    print("2. auditor.generate_audit_report('output.xlsx')")
