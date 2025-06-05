import pandas as pd


i=1
for x_rot in range(0,360,5):
    data = []
    for y_rot in range(0,360,5):
            data.append({'x_rot': str(x_rot), 'y_rot': str(y_rot), 'z_rot': '0'})

    df = pd.DataFrame(data)

    df.to_csv('./Rotations/rotations'+str(i)+'.csv', index=False)

    print('CSV file "./Rotations/rotations'+str(i)+'.csv" written successfully.')
    i+=1