# 🧠 SwiftBid AI – Product Image Category Classifier

### *(No coding required to run this project)*

This project helps you automatically identify the **category** and **sub-category** of any product just by looking at its **image**.

Example:
If you upload a shoe photo → it returns:

* **Category:** Footwear
* **Subcategory:** Running Shoes
* **Confidence:** 92%

Everything works **inside Docker**, so you **don’t need to install Python, ML tools, or any libraries** on your computer.

---

# 📌 What This Project Does

* You upload a product image (like shoes, earphones, lipstick, etc.)
* The AI model looks at the picture
* It compares it with your list of categories
* It returns the most matched **category**, **subcategory**, and **confidence score**

This is extremely useful for:

* E-commerce
* Product listing automation
* Catalog sorting
* Inventory systems

---

# 🧰 What You Need Before Starting

Only **one thing**:

✅ **Docker installed on your computer**

Nothing else.
You do NOT need:

* Python
* Machine learning knowledge
* Pytorch
* Any packages

Everything runs inside Docker.

---

# 📂 Project Structure (Simple Explanation)

```
swiftbid-clip/
 ├── app.py             → The main AI application
 ├── categories.json    → Your category & sub-category list
 ├── requirements.txt   → List of things installed inside Docker
 └── Dockerfile         → Instructions to build the Docker image
```

You only need to care about **categories.json** if you want to add or edit categories.

---

# 🚀 How To Run the Project (Step-by-Step)

## 1️⃣ Step 1 – Go inside the project folder

```
cd swiftbid-clip
```

## 2️⃣ Step 2 – Build the Docker image

(This prepares the AI environment inside Docker)

```
docker build -t swiftbid-clip .
```

This step may take a few minutes only the first time.

## 3️⃣ Step 3 – Start the AI server

```
docker run -p 8000:8000 swiftbid-clip
```

If everything is correct, you will see:

```
Uvicorn running on http://0.0.0.0:8000
```

---

# 🌐 How To Use The AI (Very Easy)

## ✅ Option 1 — Use the Browser (Easiest)

Open this link:

👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

You will see a website like this:

1. Click on **/classify**
2. Click **Try it out**
3. Upload any product image
4. Click **Execute**

You will receive the predicted category.

---

## ✅ Option 2 — Use Terminal (Optional)

If you want, you can classify an image using a command:

```
curl -X POST "http://localhost:8000/classify" \
  -F "file=@shoe.jpg"
```

Output example:

```json
{
  "category": "Footwear",
  "subcategory": "running shoes",
  "confidence": 0.92
}
```

---

# 📝 How To Add More Categories (Non-coders can do this)

Open the file:

```
categories.json
```

It looks like this:

```json
{
  "Footwear": [
    "running shoes",
    "sneakers",
    "formal shoes"
  ],
  "Electronics": [
    "laptop",
    "mobile phone",
    "smartwatch"
  ]
}
```

You can edit it like a normal list.
Just add anything you want the AI to recognize.

Example: Add “Sofa”:

```json
"Furniture": [
  "sofa",
  "chair",
  "table"
]
```

No technical knowledge required.

---

# 🧪 How Does It Work? (Simple Explanation)

* The AI model called **CLIP** understands images and text.
* When you upload a picture:

  * The model turns the image into numbers
  * Turns all category names into numbers
  * Finds which category is closest to the image meaning
* Returns the best match

You don’t need to train anything — it works out-of-the-box.
