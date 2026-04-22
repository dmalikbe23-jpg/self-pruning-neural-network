# 📄 Self-Pruning Neural Network using Learnable Gates

---

## 1. 🎯 Objective

The objective of this project is to design a neural network that can **automatically prune its own weights during training**, instead of relying on post-training pruning techniques.

This is achieved by introducing **learnable gate parameters** that control whether a weight should remain active or be removed.

---

## 2. ⚙️ Methodology

### 🔹 2.1 Prunable Layer Design

A custom layer `PrunableLinear` is implemented where:

- Each weight has an associated **gate parameter**
- Gate values are passed through a **sigmoid function**:

\[
g = \sigma(s)
\]

- The effective weight becomes:

\[
W' = W \cdot g
\]

This allows the network to:
- Keep important weights (g ≈ 1)
- Remove unimportant weights (g ≈ 0)

---

### 🔹 2.2 Model Architecture

The model consists of:

- Convolutional layers → feature extraction  
- Fully connected layers → prunable layers  

This design ensures:
- Strong feature learning  
- Efficient pruning in dense layers  

---

### 🔹 2.3 Loss Function

The total loss is defined as:

\[
Loss = CrossEntropy + \lambda \cdot SparsityLoss
\]

Where:

- **CrossEntropy Loss** → ensures classification accuracy  
- **Sparsity Loss** → L1 norm of gate values  

\[
SparsityLoss = \sum |g|
\]

---

## 3. 🧠 Why L1 Regularization?

L1 regularization is used because:

- It encourages **sparse solutions**
- It pushes many gate values toward **exact zero**
- It effectively removes redundant connections  

---

## 4. 📊 Results

| Lambda (λ) | Accuracy (%) | Sparsity (%) |
|------------|-------------|--------------|
| 1e-5       | 72.42       | 52.64        |
| 5e-5       | 72.53       | 69.44        |
| 1e-4       | 72.37       | 76.96        |

---

## 5. 📈 Observations

- Increasing λ leads to higher sparsity  
- Accuracy remains stable at moderate pruning levels  
- Very high sparsity (~77%) is achieved with minimal accuracy drop  

👉 **Best trade-off observed at λ = 1e-4**

---

## 6. 📉 Gate Distribution Analysis

The distribution of gate values shows:

- Many values near **0** → pruned weights  
- Remaining values near **1** → important connections  

This confirms that the model successfully learns to identify and remove unnecessary weights.

---

## 7. 🚀 Key Improvements

- Transitioned from basic fully connected network → CNN architecture  
- Applied pruning only to dense layers for better performance  
- Added:
  - Model checkpoint saving  
  - Automated evaluation  
  - Result logging  

---

## 8. 🎯 Conclusion

This project demonstrates that:

- Neural networks can **learn sparsity during training**
- L1 regularization effectively enforces pruning  
- Efficient and compact models can be built without post-processing pruning  

---

## 9. 🔮 Future Work

- Extend pruning to convolutional layers  
- Apply structured pruning (filters/channels)  
- Combine pruning with quantization  

---