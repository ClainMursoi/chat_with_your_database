import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import joblib
import json
import time

class MLAgent:
    def __init__(self, db_manager, llm_model):
        self.db_manager = db_manager
        self.model = llm_model
        self.cache_dir = "ml_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs("static/charts", exist_ok=True)

        self.common_targets = [
            "population", "households", "population_density",
            "enrolment", "literacy_rate",
            "health_facilities", "nurses", "disease_cases",
            "avg_income", "unemployment_rate", "gdp_estimate",
            "maize_output_tonnes", "tea_output_tonnes"
        ]

    def pretrain_all_models(self):
        print("🔄 Starting pre-training of time-series models...")
        for target in self.common_targets:
            try:
                self._pretrain_single(target)
            except Exception as e:
                print(f"   ⚠️ Skipped {target}: {e}")
        print("✅ Pre-training completed!")

    def _pretrain_single(self, target):
        for stats_table in ["county_population_stats", "county_education_stats", "county_health_stats", 
                            "county_economy_stats", "county_agriculture_stats"]:
            sql = f"SELECT year, {target} FROM {stats_table} ORDER BY year ASC"
            results, error = self.db_manager.execute_query(sql)
            if not error and len(results) >= 5:
                break
        else:
            return

        df = pd.DataFrame(results)
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df[target] = pd.to_numeric(df[target], errors='coerce')
        df = df.dropna()

        X = df['year'].values.reshape(-1, 1)
        y = df[target].values
        model = LinearRegression()
        model.fit(X, y)

        cache_path = os.path.join(self.cache_dir, f"model_{target}.joblib")
        joblib.dump(model, cache_path)
        print(f"   ✅ Pre-trained: {target} ({len(df)} records)")

    def load_or_train_model(self, target):
        cache_path = os.path.join(self.cache_dir, f"model_{target}.joblib")
        if os.path.exists(cache_path):
            return joblib.load(cache_path), False

        print(f"   → Training new model for '{target}' on-the-fly (please wait a moment)...")
        for stats_table in ["county_population_stats", "county_education_stats", "county_health_stats", 
                            "county_economy_stats", "county_agriculture_stats"]:
            sql = f"SELECT year, {target} FROM {stats_table} ORDER BY year ASC"
            results, error = self.db_manager.execute_query(sql)
            if not error and len(results) >= 5:
                break
        else:
            raise ValueError(f"Target '{target}' not found in any stats table.")

        df = pd.DataFrame(results)
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df[target] = pd.to_numeric(df[target], errors='coerce')
        df = df.dropna()

        X = df['year'].values.reshape(-1, 1)
        y = df[target].values
        model = LinearRegression()
        model.fit(X, y)

        joblib.dump(model, cache_path)
        return model, True

    def process_predictive(self, user_question, params):
        try:
            target = params.get("target", "population").lower()
            horizon = int(params.get("horizon", 5))
            county_name = params.get("county")

            print(f"\n[DEBUG] === NEW PREDICTIVE REQUEST ===")
            print(f"[DEBUG] Question: {user_question}")
            print(f"[DEBUG] Target: {target}, County: {county_name}")

            model, was_trained_now = self.load_or_train_model(target)

            # === CRITICAL: Get county_id ===
            county_id = None
            if county_name:
                print(f"[DEBUG] Looking for county: '{county_name}'")
                variants = [county_name, county_name.replace(" County", "").strip(), 
                           county_name.replace(" county", "").strip(), county_name + " County"]
                for variant in variants:
                    sql = "SELECT county_id, county_name FROM counties WHERE county_name ILIKE %s LIMIT 1"
                    result, _ = self.db_manager.execute_query(sql, (f"%{variant}%",))
                    if result and len(result) > 0:
                        county_id = result[0]['county_id']
                        real_name = result[0]['county_name']
                        print(f"[DEBUG] SUCCESS! Found county_id = {county_id} for '{real_name}'")
                        break
                if county_id is None:
                    print(f"[DEBUG] FAILED! County '{county_name}' NOT FOUND in counties table!")

            # Choose correct stats table
            if target in ["population", "households", "population_density"]:
                table = "county_population_stats"
            elif target in ["enrolment", "literacy_rate"]:
                table = "county_education_stats"
            elif target in ["health_facilities", "nurses", "disease_cases"]:
                table = "county_health_stats"
            elif target in ["avg_income", "unemployment_rate", "gdp_estimate"]:
                table = "county_economy_stats"
            else:
                table = "county_agriculture_stats"

            # Build query with county filter
            where_clause = "WHERE s.county_id = %s" if county_id is not None else ""
            values = [county_id] if county_id is not None else None

            sql = f"""
            SELECT s.year, s.{target}
            FROM {table} s
            {where_clause}
            ORDER BY s.year ASC
            """

            print(f"[DEBUG] Executing SQL: {sql} with values {values}")

            results, error = self.db_manager.execute_query(sql, values)
            if error:
                raise ValueError(f"Data fetch error: {error}")

            print(f"[DEBUG] Fetched {len(results)} rows")

            df = pd.DataFrame(results)
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df[target] = pd.to_numeric(df[target], errors='coerce')
            df = df.dropna()

            if len(df) < 5:
                raise ValueError("Not enough historical data for this prediction.")

            max_year = int(df['year'].max())
            future_years = np.arange(max_year + 1, max_year + horizon + 1).reshape(-1, 1)
            raw_preds = model.predict(future_years)

            # Realistic growth capping
            last_value = float(df[target].iloc[-1]) if len(df) > 0 else 0
            if last_value > 0:
                max_growth = 1.08
                realistic_preds = [last_value * (max_growth ** i) for i in range(1, horizon + 1)]
                predictions = list(zip(future_years.flatten().tolist(), realistic_preds))
            else:
                predictions = list(zip(future_years.flatten().tolist(), raw_preds.tolist()))

            X = df['year'].values.reshape(-1, 1)
            y = df[target].values
            y_pred = model.predict(X)
            metrics = {"mae": float(mean_absolute_error(y, y_pred)), "r2": float(r2_score(y, y_pred))}

            explanation = self.generate_explanation(df, predictions, metrics, user_question, county_name)
            chart_url = self.create_visualization(df, predictions, target)

            return {
                "historical": df.to_dict(orient="records"),
                "predictions": predictions,
                "metrics": metrics,
                "explanation": explanation,
                "chart_url": chart_url,
                "was_trained_now": was_trained_now
            }

        except Exception as e:
            raise ValueError(f"Predictive error: {str(e)}")

    def generate_explanation(self, historical_df, predictions, metrics, original_question, county_name):
        prompt = f"""
Explain this real demographic data clearly and honestly.

Historical Data: {historical_df.to_json(orient="records")}
Predictions: {json.dumps(predictions)}
Metrics: {json.dumps(metrics)}
County: {county_name or 'National level'}

Question: {original_question}

Give a short, natural explanation. Mention if the forecast is reliable or not.
"""
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def create_visualization(self, historical_df, predictions, target_col):
        unique_id = int(time.time())
        chart_path = f"static/charts/{unique_id}.png"

        plt.figure(figsize=(11, 6))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.plot(historical_df['year'], historical_df[target_col], marker='o', linewidth=2.5, label="Historical", color="#00d4ff")

        if predictions:
            future_years, pred_values = zip(*predictions)
            plt.plot(future_years, pred_values, marker='x', linestyle="--", linewidth=2.5, label="Predicted", color="#ff6b6b")

        plt.title(f"{target_col.replace('_', ' ').title()} Trend", fontsize=16)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel(target_col.replace('_', ' ').title(), fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        plt.close()
        return f"/static/charts/{unique_id}.png"