    def audit_location_similarity(self):
        """
        Detect suspiciously similar visit locations by trainer
        Enhanced: Flags trainers with any repeated/very close coordinates, not just by percentage threshold
        """
        location_flags = []

        for trainer in self.df['Visitation: Created By'].unique():
            trainer_visits = self.df[self.df['Visitation: Created By'] == trainer].copy()

            if len(trainer_visits) < 2:
                continue

            # Calculate pairwise distances between all visits
            similar_locations = []
            repeated_locations = {}

            # Track visits with identical or near-identical coordinates
            for i, row1 in trainer_visits.iterrows():
                lat1, lon1 = row1['Visit Location (Latitude)'], row1['Visit Location (Longitude)']
                if pd.isna(lat1) or pd.isna(lon1):
                    continue
                # Round coordinates for grouping (5 decimals ~1 meter)
                coord_key = (round(lat1, 5), round(lon1, 5))
                repeated_locations.setdefault(coord_key, []).append(i)

            # Add all groups with >1 visit (i.e., repeated location)
            for coord, idxs in repeated_locations.items():
                if len(idxs) > 1:
                    # Flag all pairs in the repeated location
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

            # Also keep original logic for visits within 50 meters
            for i, row1 in trainer_visits.iterrows():
                for j, row2 in trainer_visits.iterrows():
                    if i >= j:
                        continue
                    lat1, lon1 = row1['Visit Location (Latitude)'], row1['Visit Location (Longitude)']
                    lat2, lon2 = row2['Visit Location (Latitude)'], row2['Visit Location (Longitude)']
                    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
                        continue
                    distance = geodesic((lat1, lon1), (lat2, lon2)).meters
                    if distance < 50:
                        similar_locations.append({
                            'visit_1_index': i,
                            'visit_2_index': j,
                            'distance_meters': distance,
                            'accounts': f"{row1['Account Name']} & {row2['Account Name']}"
                        })

            # Flag the trainer if any suspicious pairs are found
            if similar_locations:
                location_flags.append({
                    'trainer': trainer,
                    'similarity_percentage': len(similar_locations) / max(1, (len(trainer_visits) * (len(trainer_visits) - 1) / 2)) * 100,
                    'similar_locations': similar_locations,
                    'total_visits': len(trainer_visits),
                    'flagged_pairs': len(similar_locations)
                })

        return location_flags
