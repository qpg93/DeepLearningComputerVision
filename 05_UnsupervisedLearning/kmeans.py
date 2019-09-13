import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

def assignment(df, centroids, colmap):
    # Calculate distance
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = (
            np.sqrt( # sqrt((x1 - x2)^2 - (y1 - y2)^2)
                (df['x'] - centroids[i][0])**2 + (df['y'] - centroids[i][1])**2
                )
            )
    # print(df)

    # Give color
    distance_from_cenroid_id = ['distance_from_{}'.format(i) for i in centroids.keys()]
    # print(distance_from_cenroid_id)
    df['closest'] = df.loc[:, distance_from_cenroid_id].idxmin(axis=1) # idxmin return index of first occurrence of minimum over requested axis
    # print(df['closest'])
    df['closest'] = df['closest'].map(lambda x : int(x.lstrip('distance_from_'))) # lstrip() removes any leading characters
    # print(df['closest'])
    df['color'] = df['closest'].map(lambda x : colmap[x])
    # print(df['color'])
    # print(df)
    return df

def update(df, centroids):
    # Recalculate centroids
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids

def main():
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
        })

    k = 3
    # Randomly choose k centroids
    centroids = {
        i: [np.random.randint(0, 80), np.random.randint(0, 80)] for i in range(k)
    }

    colmap = {0: 'r', 1: 'g', 2: 'b'}
    df = assignment(df, centroids, colmap)

    # Draw all points
    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolors='k')
    # Draw centroids
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.show()

    for i in range(10):
        plt.close() # Close previous plot

        closest_centroids = df['closest'].copy(deep=True)
        centroids = update(df, centroids)
        plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolors='k')
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.show()

        df = assignment(df, centroids, colmap)

        if closest_centroids.equals(df['closest']):
            break

if __name__ == '__main__':
    main()