import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_session(skill="expert", duration_s=30, hz=100):
    n = duration_s * hz
    t = np.linspace(0, duration_s, n)

    if skill == "expert":
        travel_speed   = 8.0  + np.random.normal(0, 0.3, n)
        work_angle     = 90.0 + np.random.normal(0, 1.5, n)
        travel_angle   = 15.0 + np.random.normal(0, 1.0, n)
        arc_length     = 6.0  + np.random.normal(0, 0.4, n)
        tremor         = np.abs(np.random.normal(0, 0.02, n))
        voltage        = 22.0 + np.random.normal(0, 0.3, n)
        current        = 180.0+ np.random.normal(0, 2.0, n)
        wfs            = 5.0  + np.random.normal(0, 0.1, n)
        label          = np.zeros(n, dtype=int)
    else:
        travel_speed   = 8.0  + np.random.normal(0, 1.8, n)
        work_angle     = 90.0 + np.random.normal(0, 6.0, n)
        travel_angle   = 15.0 + np.random.normal(0, 5.0, n)
        arc_length     = 6.0  + np.random.normal(0, 2.0, n)
        tremor         = np.abs(np.random.normal(0, 0.12, n))
        voltage        = 22.0 + np.random.normal(0, 1.2, n)
        current        = 180.0+ np.random.normal(0, 12.0, n)
        wfs            = 5.0  + np.random.normal(0, 0.6, n)
        label          = np.zeros(n, dtype=int)

        for _ in range(4):
            if n - hz * 4 <= hz * 2:
                continue
            start = np.random.randint(hz * 2, n - hz * 4)
            length = np.random.randint(hz, hz * 3)
            end = min(start + length, n)
            error_type = np.random.choice(["angle", "speed", "arc"])
            if error_type == "angle":
                work_angle[start:end]   += np.random.choice([-1, 1]) * np.random.uniform(15, 30)
                travel_angle[start:end] += np.random.choice([-1, 1]) * np.random.uniform(10, 20)
            elif error_type == "speed":
                travel_speed[start:end] += np.random.choice([-1, 1]) * np.random.uniform(4, 8)
            else:
                arc_length[start:end]   += np.random.uniform(3, 8)
            label[start:end] = 1

    x = travel_speed.cumsum() / hz
    y = np.random.normal(0, 0.5, n).cumsum() * 0.01
    z = np.random.normal(0, 0.2, n).cumsum() * 0.005

    gyro_x = np.gradient(work_angle,   1/hz)
    gyro_y = np.gradient(travel_angle, 1/hz)
    gyro_z = np.gradient(arc_length,   1/hz) * 0.1

    accel_x = np.gradient(travel_speed, 1/hz)
    accel_y = np.random.normal(0, 0.1 if skill=="expert" else 0.5, n)
    accel_z = np.random.normal(0, 0.1 if skill=="expert" else 0.5, n)

    df = pd.DataFrame({
        "t":            t,
        "skill":        skill,
        "x":            x,
        "y":            y,
        "z":            z,
        "roll":         work_angle,
        "pitch":        travel_angle,
        "yaw":          arc_length * 2,
        "gyro_x":       gyro_x,
        "gyro_y":       gyro_y,
        "gyro_z":       gyro_z,
        "accel_x":      accel_x,
        "accel_y":      accel_y,
        "accel_z":      accel_z,
        "work_angle":   work_angle,
        "travel_angle": travel_angle,
        "travel_speed": travel_speed,
        "arc_length":   arc_length,
        "tremor":       tremor,
        "voltage":      voltage,
        "current":      current,
        "wfs":          wfs,
        "error":        label,
    })
    return df


if __name__ == "__main__":
    # Works on Windows, Mac, and Linux
    output_dir = os.path.join("data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "welding_data.csv")

    sessions = []
    for i in range(5):
        sessions.append(generate_session("expert",   duration_s=30))
        sessions.append(generate_session("beginner", duration_s=30))

    df = pd.concat(sessions, ignore_index=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} frames to {output_path}")
    print(f"Error frames: {df['error'].sum():,} ({df['error'].mean()*100:.1f}%)")
    print(df[["work_angle","travel_speed","arc_length","tremor","voltage","error"]].describe().round(2))