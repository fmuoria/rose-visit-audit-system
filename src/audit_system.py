from geopy.distance import geodesic
import pandas as pd

class VisitAuditSystem:
    def __init__(self, df):
        self.df = df

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

            # Check for "too close" visits (<150m apart)
            for i, row1 in trainer_visits.iterrows():
                for j, row2 in trainer_visits.iterrows():
                    if i >= j:
                        continue
                    lat1, lon1 = row1['Visit Location (Latitude)'], row1['Visit Location (Longitude)']
                    lat2, lon2 = row2['Visit Location (Latitude)'], row2['Visit Location (Longitude)']
                    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
                        continue
                    distance = geodesic((lat1, lon1), (lat2, lon2)).meters
                    if distance < 150:
                        similar_locations.append({
                            'visit_1_index': i,
                            'visit_2_index': j,
                            'distance_meters': distance,
                            'accounts': f"{row1['Account Name']} & {row2['Account Name']}"
                        })

            # Collect all flags
            if similar_locations or out_of_country:
                location_flags.append({
                    'trainer': trainer,
                    'similarity_percentage': len(similar_locations) / max(1, (len(trainer_visits) * (len(trainer_visits) - 1) / 2)) * 100,
                    'similar_locations': similar_locations,
                    'out_of_country': out_of_country,
                    'total_visits': len(trainer_visits),
                    'flagged_pairs': len(similar_locations),
                    'flagged_out_of_country': len(out_of_country)
                })

        return location_flags
