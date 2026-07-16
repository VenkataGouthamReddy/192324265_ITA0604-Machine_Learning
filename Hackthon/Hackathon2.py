from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# -----------------------------
# Customer Purchase Dataset
# Features:
# [Electronics, Clothing, Grocery]
# -----------------------------

X = [
    [1,0,0],   # Electronics
    [1,1,0],   # Electronics + Clothing
    [0,1,0],   # Clothing
    [0,1,1],   # Clothing + Grocery
    [0,0,1],   # Grocery
    [1,0,1],   # Electronics + Grocery
    [1,1,1],   # All Categories
    [0,0,0]    # New Customer
]

# Recommended Products
y = [
    "Mouse",
    "Headphones",
    "Shoes",
    "Bag",
    "Snacks",
    "Keyboard",
    "Smart Watch",
    "Welcome Offer"
]

# -----------------------------
# Train Decision Tree
# -----------------------------

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

model.fit(X, y)

# -----------------------------
# Visualize Decision Tree
# -----------------------------

plt.figure(figsize=(14,8))

plot_tree(
    model,
    feature_names=[
        "Electronics",
        "Clothing",
        "Grocery"
    ],
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=10,
    impurity=False
)

plt.title(
    "E-Commerce Product Recommendation System\nDecision Tree",
    fontsize=16,
    fontweight="bold"
)

plt.savefig("Product_Recommendation_Tree.png", dpi=300)
plt.show()

# -----------------------------
# Recommendation
# -----------------------------

customer = [[1,1,0]]

prediction = model.predict(customer)

print("========= CUSTOMER DETAILS =========")
print("Purchased Electronics : Yes")
print("Purchased Clothing    : Yes")
print("Purchased Grocery     : No")

print("\nRecommended Product :", prediction[0])

print("\nDecision Tree saved as Product_Recommendation_Tree.png")