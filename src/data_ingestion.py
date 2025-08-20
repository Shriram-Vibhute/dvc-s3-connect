import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pathlib

# Generating paths
root_dir = pathlib.Path(__file__).parent.parent
data_path = root_dir / "data" / "raw" / "students.csv"

df = pd.read_csv(data_path)

# Separating features and target variable
X = df.drop(columns=['Placed'])
y = df['Placed']

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Creating a DataFrame with PCA results
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['Placed'] = y.values

df_pca.to_csv(root_dir / "data" / "processed" / "student_performance_pca.csv", index=False)