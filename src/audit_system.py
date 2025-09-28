"""
ROSE Women Leaders Visit Audit System
Automated audit system for detecting suspicious patterns in beneficiary visits
WITH IMPROVED KENYA LOCATION CHECKING
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
        # Convert similarity threshold to meters for location checking (0.7 = ~150m)
        self.location_threshold_meters = int(similarity_threshold * 200)  # Scale to reasonable meters
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
        Detect suspiciously similar or anomalous visit locations by trainer
        - Flags repeated/very close coordinates
        - Flags coordinates outside Kenya bounding box
        """
        location_flags = []

        # Define Kenya bounding box (approx)
        kenya_lat_min, kenya_lat_max = -5, 5
        kenya_lon_min, kenya_lon_max = 33, 42

        for trainer in self.df['Visitation: Created By'].unique():
            trainer_visits = self.df[self.df['Visitation: Created By'] == trainer].copy()

            if len(trainer_visits) < 2:
                continue

            similar_locations = []
            repeated_locations = {}
            out_of_country = []

            # Check for out-of-country anomalies
            for i, row in trainer_visits.iterrows():
                lat, lon = row['Visit Location (Latitude)'], row['Visit Location (Longitude)']
                if pd.isna(lat) or pd.isna(lon):
                    continue
                if not (kenya_lat_min <= lat <= kenya_lat_max and kenya_lon_min <= lon <= kenya_lon_max):
                    out_of_country.append({
                        'visit_index': i,
                        'account': row['Account Name'],
                        'coordinates': (lat, lon),
                        'reason': "Outside Kenya"
                    })

            # Track repeated coordinates (identical within ~1m)
            for i, row1 in trainer_visits.iterrows():
                lat1, lon1 = row1['Visit Location (Latitude)'], row1['Visit Location (Longitude)']
                if pd.isna(lat1) or pd.isna(lon1):
                    continue
                coord_key = (round(lat1, 5), round(lon1, 5))
                repeated_locations.setdefault(coord_key, []).append(i)

            for coord, idxs in repeated_locations.items():
                if len(idxs) > 1:
                    for i in range(len(idxs)):
                        for j in range(i + 1, len(idxs)):
                            row1 = trainer_visits.loc[idxs[i]]
                            row2 = trainer_visits.loc[idxs[j]]
                            similar_locations.append({
                                'visit_1_index': idxs[i],
                                'visit_2_index': idxs[j],
                                'distance_meters': 0,
                                'accounts': f"{row1['Account Name']} & {row2['Account Name']}",
                                'coordinates': coord
                            })

            # Check for "too close" visits (<location_threshold_meters apart)
            for i, row1 in trainer_visits.iterrows():
                for j, row2 in trainer_visits.iterrows():
                    if i >= j:
                        continue
                    lat1, lon1 = row1['Visit Location (Latitude)'], row1['Visit Location (Longitude)']
                    lat2, lon2 = row2['Visit Location (Latitude)'], row2['Visit Location (Longitude)']
                    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
                        continue
                    distance = geodesic((lat1, lon1), (lat2, lon2)).meters
                    if distance < self.location_threshold_meters:
                        similar_locations.append({
                            'visit_1_index': i,
                            'visit_2_index': j,
                            'distance_meters': distance,
                            'accounts': f"{row1['Account Name']} & {row2['Account Name']}"
                        })

            # Collect all flags
            if similar_locations or out_of_country:
                total_comparisons = len(trainer_visits) * (len(trainer_visits) - 1) / 2
                location_flags.append({
                    'trainer': trainer,
                    'similarity_percentage': len(similar_locations) / max(1, total_comparisons),
                    'similar_locations': similar_locations,
                    'out_of_country': out_of_country,
                    'total_visits': len(trainer_visits),
                    'flagged_pairs': len(similar_locations),
                    'flagged_out_of_country': len(out_of_country)
                })

        return location_flags
    
    def audit_story_similarity(self):
        """
        Enhanced story similarity detection to catch trainers using generic responses
        - Flags trainers with consistently similar stories (not engaging beneficiaries)
        - Flags very short stories (lack of engagement)
        - Uses stricter thresholds to catch generic copy-paste behavior
        """
        story_flags = []
        
        for trainer in self.df['Visitation: Created By'].unique():
            trainer_visits = self.df[self.df['Visitation: Created By'] == trainer].copy()
            stories = trainer_visits['Story Summary'].tolist()
            
            if len(stories) < 2:
                continue
            
            # Clean and analyze all stories (including short ones for flagging)
            all_stories = [str(s).strip() for s in stories if pd.notna(s)]
            
            if len(all_stories) < 2:
                continue
            
            # Flag very short stories (lack of engagement)
            short_stories = [s for s in all_stories if len(s.split()) < 15]  # Less than 15 words
            very_short_stories = [s for s in all_stories if len(s) < 50]  # Less than 50 characters
            
            # Only analyze stories with meaningful content for similarity
            meaningful_stories = [s for s in all_stories if len(s.split()) >= 10]  # At least 10 words
            
            similar_stories = []
            avg_similarity = 0
            story_diversity = 1
            
            if len(meaningful_stories) >= 2:
                # Calculate TF-IDF similarity matrix with enhanced settings for catching generic text
                vectorizer = TfidfVectorizer(
                    stop_words='english',
                    ngram_range=(1, 3),  # Include 3-grams to catch phrase patterns
                    max_features=500,    # Reduced to focus on key terms
                    lowercase=True,
                    min_df=1,           # Include all terms
                    max_df=0.9          # Exclude very common terms
                )
                
                try:
                    tfidf_matrix = vectorizer.fit_transform(meaningful_stories)
                    similarity_matrix = cosine_similarity(tfidf_matrix)
                    
                    # Find similar story pairs with lower threshold to catch subtle similarities
                    for i in range(len(similarity_matrix)):
                        for j in range(i + 1, len(similarity_matrix)):
                            similarity_score = similarity_matrix[i][j]
                            
                            # Lower threshold for detecting generic responses
                            if similarity_score >= 0.4:  # 40% similarity threshold
                                similar_stories.append({
                                    'story_1_index': i,
                                    'story_2_index': j,
                                    'similarity_score': similarity_score,
                                    'story_1_preview': meaningful_stories[i][:120] + '...',
                                    'story_2_preview': meaningful_stories[j][:120] + '...'
                                })
                    
                    # Calculate overall metrics
                    upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
                    avg_similarity = np.mean(upper_triangle)
                    
                    # Enhanced story diversity check
                    story_diversity = len(set(meaningful_stories)) / len(meaningful_stories)
                    
                except Exception as e:
                    print(f"Error processing stories for trainer {trainer}: {e}")
            
            # Calculate story lengths and engagement metrics
            story_lengths = [len(s.split()) for s in all_stories]
            avg_story_length = np.mean(story_lengths) if story_lengths else 0
            
            # Enhanced authenticity scoring
            authenticity_score = self._calculate_enhanced_story_authenticity(
                avg_similarity, story_diversity, avg_story_length, 
                len(short_stories), len(very_short_stories), len(all_stories)
            )
            
            # Flag if authenticity is low OR there are engagement issues
            short_story_percentage = len(short_stories) / len(all_stories) if all_stories else 0
            very_short_percentage = len(very_short_stories) / len(all_stories) if all_stories else 0
            
            # More aggressive flagging criteria
            should_flag = (
                authenticity_score < 80 or  # Lower threshold for authenticity
                avg_similarity > 0.3 or     # Lower threshold for average similarity
                story_diversity < 0.7 or    # Higher threshold for diversity
                short_story_percentage > 0.3 or  # More than 30% short stories
                very_short_percentage > 0.2 or   # More than 20% very short stories
                len(similar_stories) > 0
            )
            
            if should_flag:
                story_flags.append({
                    'trainer': trainer,
                    'average_similarity': avg_similarity,
                    'story_diversity': story_diversity,
                    'average_length': avg_story_length,
                    'similar_stories': similar_stories,
                    'total_stories': len(all_stories),
                    'short_stories_count': len(short_stories),
                    'very_short_stories_count': len(very_short_stories),
                    'short_story_percentage': short_story_percentage,
                    'very_short_percentage': very_short_percentage,
                    'authenticity_score': authenticity_score,
                    'engagement_issues': {
                        'generic_responses': len(similar_stories) > 0,
                        'too_many_short': short_story_percentage > 0.3,
                        'too_many_very_short': very_short_percentage > 0.2,
                        'low_diversity': story_diversity < 0.7,
                        'high_similarity': avg_similarity > 0.3
                    }
                })
        
        return story_flags
    
    def _calculate_enhanced_story_authenticity(self, avg_similarity, story_diversity, avg_length, 
                                             short_count, very_short_count, total_stories):
        """
        Enhanced story authenticity score focusing on genuine engagement
        (0-100, higher is more authentic/engaged)
        """
        score = 100  # Start with perfect score
        
        # Heavy penalty for high similarity (generic responses)
        if avg_similarity > 0.5:
            score -= 40
        elif avg_similarity > 0.3:
            score -= 25
        elif avg_similarity > 0.2:
            score -= 10
        
        # Heavy penalty for low diversity (copy-paste behavior)
        if story_diversity < 0.5:
            score -= 35
        elif story_diversity < 0.7:
            score -= 20
        elif story_diversity < 0.85:
            score -= 10
        
        # Penalty for short stories (lack of engagement)
        short_percentage = short_count / total_stories if total_stories > 0 else 0
        very_short_percentage = very_short_count / total_stories if total_stories > 0 else 0
        
        if very_short_percentage > 0.3:
            score -= 30
        elif very_short_percentage > 0.2:
            score -= 20
        elif very_short_percentage > 0.1:
            score -= 10
            
        if short_percentage > 0.5:
            score -= 25
        elif short_percentage > 0.3:
            score -= 15
        
        # Length-based scoring (reward detailed engagement)
        if avg_length < 10:  # Very short average
            score -= 20
        elif avg_length < 20:  # Short average
            score -= 10
        elif avg_length > 50:  # Good detail
            score += 5
        elif avg_length > 80:  # Excellent detail
            score += 10
        
        return max(0, min(100, score))
    
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
        Calculate confidence score for each individual visit with detailed explanations
        High standard: 90%+ is good, anything below needs verification
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
            issues_found = []
            recommendations = []
            
            # Location confidence penalty
            if trainer in location_flags:
                flag_data = location_flags[trainer]
                penalty = flag_data['similarity_percentage'] * 40  # Increased penalty
                confidence_score -= penalty
                if penalty > 1:
                    issues_found.append(f"Location similarity concerns (-{penalty:.1f})")
                    if flag_data.get('out_of_country'):
                        recommendations.append("Verify GPS coordinates and actual visit location")
                    else:
                        recommendations.append("Confirm visit was at beneficiary's actual business location")
            
            # Story engagement penalty (enhanced)
            if trainer in story_flags:
                flag_data = story_flags[trainer]
                story_authenticity = flag_data['authenticity_score']
                penalty = (100 - story_authenticity) * 0.5  # Increased penalty
                confidence_score -= penalty
                
                if penalty > 1:
                    engagement_issues = flag_data.get('engagement_issues', {})
                    issue_details = []
                    
                    if engagement_issues.get('generic_responses'):
                        issue_details.append("generic/similar responses")
                        recommendations.append("Verify specific discussion topics with beneficiary")
                    
                    if engagement_issues.get('too_many_short') or engagement_issues.get('too_many_very_short'):
                        issue_details.append("very brief stories indicating minimal engagement")
                        recommendations.append("Confirm trainer spent adequate time with beneficiary")
                    
                    if engagement_issues.get('low_diversity'):
                        issue_details.append("repetitive content across visits")
                        recommendations.append("Ask beneficiary about specific advice/training received")
                    
                    issues_found.append(f"Story engagement concerns: {', '.join(issue_details)} (-{penalty:.1f})")
            
            # Upload delay penalty
            upload_delay = visit.get('Upload Delay Days', 0)
            if upload_delay > self.upload_delay_threshold:
                delay_penalty = min((upload_delay / self.upload_delay_threshold) * 25, 50)  # Increased penalty
                confidence_score -= delay_penalty
                issues_found.append(f"Late upload: {upload_delay:.0f} days after visit (-{delay_penalty:.1f})")
                recommendations.append("Verify visit actually occurred on the recorded date")
            
            # Income anomaly penalty
            if trainer in income_flags:
                flag_data = income_flags[trainer]
                income_suspicion = flag_data['suspicious_score']
                penalty = income_suspicion * 0.25  # Increased penalty
                confidence_score -= penalty
                
                if penalty > 1:
                    issues_found.append(f"Income reporting concerns (-{penalty:.1f})")
                    recommendations.append("Verify income figures and request supporting documentation")
            
            # Temporal clustering penalty
            if trainer in temporal_flags:
                flag_data = temporal_flags[trainer]
                temporal_suspicion = flag_data['temporal_suspicion_score']
                penalty = temporal_suspicion * 0.2  # Increased penalty
                confidence_score -= penalty
                
                if penalty > 1:
                    issues_found.append(f"Visit timing concerns (-{penalty:.1f})")
                    recommendations.append("Confirm actual visit date and time")
            
            # Data completeness penalty (stricter)
            required_fields = ['Account Name', 'Story Summary', 'Visit Location (Latitude)', 
                             'Visit Location (Longitude)', 'Current Monthly Income']
            complete_fields = sum(1 for field in required_fields if pd.notna(visit[field]) and visit[field] != '')
            
            if complete_fields < len(required_fields):
                missing_count = len(required_fields) - complete_fields
                completeness_penalty = missing_count * 8  # 8 points per missing field
                confidence_score -= completeness_penalty
                issues_found.append(f"Missing {missing_count} required field(s) (-{completeness_penalty:.1f})")
                recommendations.append("Complete all required visit documentation")
            
            # Story length check (new)
            story_text = str(visit.get('Story Summary', ''))
            story_word_count = len(story_text.split()) if story_text else 0
            
            if story_word_count < 15:  # Less than 15 words
                story_penalty = 15
                confidence_score -= story_penalty
                issues_found.append(f"Very brief story ({story_word_count} words, -{story_penalty})")
                recommendations.append("Verify meaningful engagement occurred during visit")
            elif story_word_count < 25:  # Less than 25 words
                story_penalty = 8
                confidence_score -= story_penalty
                issues_found.append(f"Brief story ({story_word_count} words, -{story_penalty})")
                recommendations.append("Confirm adequate discussion and training provided")
            
            # Determine final assessment with 90% threshold
            final_score = max(0, min(100, confidence_score))
            phone_number = visit.get('Phone', '')
            
            if final_score >= 90:
                assessment = "HIGH CONFIDENCE"
                display_phone = ""  # No phone needed
                call_recommended = "No"
            elif final_score >= 75:
                assessment = "MEDIUM CONFIDENCE - Verification Recommended"
                display_phone = phone_number if phone_number else "Phone not available"
                call_recommended = "Yes - Recommended"
            else:
                assessment = "LOW CONFIDENCE - Requires Immediate Verification"
                display_phone = phone_number if phone_number else "Phone not available"
                call_recommended = "Yes - Priority"
            
            # Create comprehensive explanation
            if not issues_found:
                explanation = "No issues detected - all checks passed with high confidence"
            else:
                explanation = "; ".join(issues_found)
            
            # Create recommendation summary
            if not recommendations:
                recommendation_text = "No additional verification needed"
            else:
                unique_recommendations = list(set(recommendations))
                recommendation_text = "; ".join(unique_recommendations)
            
            visit_scores.append({
                'visit_index': visit.name,
                'trainer': trainer,
                'account_name': visit['Account Name'],
                'visit_date': visit['Visitation Date'],
                'confidence_score': final_score,
                'confidence_assessment': assessment,
                'issues_found': explanation,
                'verification_needed': call_recommended,
                'verification_recommendations': recommendation_text,
                'beneficiary_phone': display_phone,
                'upload_delay_days': upload_delay,
                'story_word_count': story_word_count,
                'program_cohort': visit.get('Program Cohort', ''),
                'business_sector': visit.get('Business Sector', ''),
                'visit_location': f"({visit.get('Visit Location (Latitude)', 'N/A')}, {visit.get('Visit Location (Longitude)', 'N/A')})"
            })
        
        return visit_scores
    
    def calculate_trainer_confidence(self):
        """
        Calculate overall confidence score for each trainer with 90% standard
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
            # Count visits below 90% (new standard)
            below_90_count = sum(1 for s in scores if s < 90)
            # Count visits needing verification (below 75%)
            needs_verification_count = sum(1 for s in scores if s < 75)
            
            trainer_confidence.append({
                'trainer': trainer,
                'average_confidence': avg_confidence,
                'minimum_confidence': min_confidence,
                'visits_below_90_percent': below_90_count,
                'visits_needing_verification': needs_verification_count,
                'total_visits': len(scores),
                'below_90_percentage': (below_90_count / len(scores)) * 100,
                'verification_percentage': (needs_verification_count / len(scores)) * 100,
                'audit_priority': self._calculate_audit_priority(avg_confidence, below_90_count, len(scores))
            })
        
        return sorted(trainer_confidence, key=lambda x: x['average_confidence'])
    
    def _calculate_audit_priority(self, avg_confidence, low_conf_count, total_visits):
        """
        Calculate audit priority with 90% confidence standard
        """
        below_90_percentage = low_conf_count / total_visits if total_visits > 0 else 0
        
        if avg_confidence < 70 or below_90_percentage > 0.5:
            return "HIGH"
        elif avg_confidence < 85 or below_90_percentage > 0.3:
            return "MEDIUM"  
        else:
            return "LOW"

# Example usage
if __name__ == "__main__":
    # Initialize audit system
    auditor = VisitAuditSystem(
        similarity_threshold=0.7,  # 70% similarity threshold
        upload_delay_threshold=7   # 7 days upload delay threshold
    )
    
    # Example of how to use with your data
    # auditor.load_data('visit_data.csv')
    # results would be generated here
    
    print("ROSE Visit Audit System Ready!")
    print("Usage:")
    print("1. auditor.load_data('your_data.csv')")
    print("2. location_flags = auditor.audit_location_similarity()")
    print("3. visit_scores = auditor.calculate_visit_confidence()")
