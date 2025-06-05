import pandas as pd

data = []
for x_rot in range(0,360,5):
    for y_rot in range(0,360,5):
            data.append({'x_rot': str(x_rot), 'y_rot': str(y_rot), 'z_rot': '0'})

df = pd.DataFrame(data)

df.to_csv('rotations.csv', index=False)

print("CSV file 'rotations.csv' written successfully.")