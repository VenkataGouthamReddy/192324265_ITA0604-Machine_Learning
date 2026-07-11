from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# -----------------------------
# Dataset
# -----------------------------
# Features: [Rainfall, Temperature, Humidity, Nitrogen]
X = [
    [202, 20, 82, 90],   # Rice
    [210, 22, 80, 85],   # Rice
    [80, 30, 55, 25],    # Maize
    [90, 29, 60, 30],    # Maize
    [120, 26, 65, 40],   # Wheat
    [110, 24, 68, 45],   # Wheat
    [60, 32, 45, 15],    # Cotton
    [65, 33, 48, 18]     # Cotton
]

# Crop Labels
y = [
    "Rice",
    "Rice",
    "Maize",
    "Maize",
    "Wheat",
    "Wheat",
    "Cotton",
    "Cotton"
]

# -----------------------------
# Train Decision Tree
# -----------------------------
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

model.fit(X, y)

# -----------------------------
# Plot Professional Tree
# -----------------------------
plt.figure(figsize=(18,10), facecolor="white")

plot_tree(
    model,
    feature_names=[
        "Rainfall",
        "Temperature",
        "Humidity",
        "Nitrogen"
    ],
    class_names=model.classes_,
    filled=True,          # Colorful nodes
    rounded=True,         # Rounded boxes
    fontsize=11,
    impurity=False,       # Remove Gini values
    proportion=False,
    precision=1
)

plt.title(
    "SMART CROP RECOMMENDATION SYSTEM\nDecision Tree Classification",
    fontsize=18,
    fontweight="bold",
    color="darkblue",
    pad=20
)

plt.tight_layout()

plt.savefig(
    "Smart_Crop_Decision_Tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Decision tree saved successfully as Smart_Crop_Decision_Tree.png")

print("\nNode Types:")
print("Root Node      : First node of the tree")
print("Decision Node  : Splits data based on a feature")
print("Leaf Node      : Final predicted crop")