import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

label = 'neosonrasıpathboyutu'

data = pd.read_excel(f"dataset/input_excels/{label}_balanced_data.xlsx")

# input and output columns
features = [
    'yas', 'neoboyutUSG', 'neooncesiaksilla', 'kliniklenfkat',
    'estkat', 'progkat2', 'cerb2', 'ki67pre',
    'preopmetastazvarlığı', 'KLİNİKLAP', 'USGLAP', 'MRLAP', 'ÇAP', 'SAYI'
]
output = label

X = data[features]
y = data[[output]]

# 2. one-hot encoding for categorical features
categorical_features = ['KLİNİKLAP', 'USGLAP', 'MRLAP']
onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_categories = onehot_encoder.fit_transform(X[categorical_features])
encoded_category_names = onehot_encoder.get_feature_names_out(categorical_features)

X_encoded = pd.DataFrame(encoded_categories, columns=encoded_category_names)

# scaling for numeric features
numeric_features = X.drop(columns=categorical_features)
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(numeric_features)
X_scaled = pd.DataFrame(scaled_numeric, columns=numeric_features.columns)

# concat input
X_final = pd.concat([X_scaled, X_encoded], axis=1)

# one-hot encoding the output
onehot_output = OneHotEncoder(sparse_output=False)
y_encoded = onehot_output.fit_transform(y)

# turn class labels into string
class_labels = [str(label) for label in onehot_output.categories_[0]]

# k-fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=7)

# performance monitoring
fold_accuracies = []
fold_losses = []

all_true_classes = []
all_pred_classes = []

fold_no = 1
for train_index, val_index in kfold.split(X_final):
    X_train, X_val = X_final.iloc[train_index], X_final.iloc[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]

    # model structure
    model = Sequential()
    model.add(Dense(292, input_dim=X_train.shape[1], activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(Dense(len(class_labels), activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # early callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)

    # model training
    print(f"\nFold {fold_no} - Model Training....")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=1
    )

    # PERFORMANCE TEST
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Fold {fold_no} - Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    fold_accuracies.append(val_accuracy)
    fold_losses.append(val_loss)

    # predictions
    y_val_pred = model.predict(X_val)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    y_val_true_classes = np.argmax(y_val, axis=1)

    # save predicted and true classes
    all_true_classes.extend(y_val_true_classes)
    all_pred_classes.extend(y_val_pred_classes)

    fold_no += 1

# classification report for all folds
print("\nClassification Report:")
print(classification_report(all_true_classes, all_pred_classes, target_names=class_labels))

# confusion matrix
cm = confusion_matrix(all_true_classes, all_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# model.save('models/pathological_size_model.h5')

# print cross validation ouputs
print("\nCross-Validation Results:")
print(f"Average Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"Average Loss: {np.mean(fold_losses):.4f}")
