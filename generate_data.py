import numpy as np
import pandas as pd

def generate_produce_dataset(n_samples: int = 6000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    produce_profiles = {
        "Leafy Greens": {"base_life": 10.0, "ideal_temp": 2.0, "temp_sensitivity": 0.55, "ethylene_sensitive": True},
        "Tomato":       {"base_life": 14.0, "ideal_temp": 12.0, "temp_sensitivity": 0.35, "ethylene_sensitive": True},
        "Banana":       {"base_life": 12.0, "ideal_temp": 13.0, "temp_sensitivity": 0.40, "ethylene_sensitive": True},
        "Apple":        {"base_life": 30.0, "ideal_temp": 3.0, "temp_sensitivity": 0.20, "ethylene_sensitive": False},
        "Strawberry":   {"base_life": 7.0,  "ideal_temp": 1.0, "temp_sensitivity": 0.65, "ethylene_sensitive": False},
        "Potato":       {"base_life": 45.0, "ideal_temp": 10.0, "temp_sensitivity": 0.15, "ethylene_sensitive": False},
    }
    
    produce_types = list(produce_profiles.keys())
    packaging_types = ["Corrugated Box", "Perforated Poly", "Open Crate"]
    ethylene_levels = ["Low", "Moderate", "High"]
    
    chosen_produce = np.random.choice(produce_types, size=n_samples)
    chosen_packaging = np.random.choice(packaging_types, size=n_samples)
    chosen_ethylene = np.random.choice(ethylene_levels, size=n_samples, p=[0.55, 0.30, 0.15])
    
    initial_maturity = np.random.randint(1, 6, size=n_samples)
    storage_temp = np.round(np.random.uniform(0.5, 26.0, size=n_samples), 1)
    relative_humidity = np.round(np.random.uniform(50.0, 98.0, size=n_samples), 1)
    transit_days = np.random.randint(1, 7, size=n_samples)
    
    shelf_life = []
    for i in range(n_samples):
        item = chosen_produce[i]
        prof = produce_profiles[item]
        
        temp_delta = max(0.0, storage_temp[i] - prof["ideal_temp"])
        temp_decay = temp_delta * prof["temp_sensitivity"]
        humidity_penalty = max(0.0, (85.0 - relative_humidity[i]) * 0.05)
        eth_multiplier = {"Low": 0.0, "Moderate": 1.5, "High": 3.5}[chosen_ethylene[i]]
        ethylene_decay = eth_multiplier * (1.8 if prof["ethylene_sensitive"] else 0.4)
        maturity_decay = (initial_maturity[i] - 1) * 1.5
        transit_decay = transit_days[i] * 0.9
        
        total_loss = temp_decay + humidity_penalty + ethylene_decay + maturity_decay + transit_decay
        calculated_life = prof["base_life"] - total_loss + np.random.normal(0, 0.5)
        shelf_life.append(round(max(0.2, calculated_life), 1))
        
    return pd.DataFrame({
        "produce_type": chosen_produce,
        "initial_maturity": initial_maturity,
        "storage_temp": storage_temp,
        "relative_humidity": relative_humidity,
        "ethylene_exposure": chosen_ethylene,
        "transit_days": transit_days,
        "packaging_type": chosen_packaging,
        "shelf_life_days": shelf_life
    })

if __name__ == "__main__":
    dataset = generate_produce_dataset()
    dataset.to_csv("produce_shelf_life.csv", index=False)
    print(f"Generated {dataset.shape[0]} records in produce_shelf_life.csv")
