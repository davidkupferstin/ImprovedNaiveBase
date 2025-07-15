const BACKEND_URL = "http://127.0.0.1:8500" // ודא שזו הכתובת הנכונה של שרת ה-FastAPI שלך

// אלמנטים ב-DOM
const uploadForm = document.getElementById("uploadForm")
const csvFile = document.getElementById("csvFile")
const uploadStatus = document.getElementById("uploadStatus")
const modelStatusDiv = document.getElementById("modelStatus")
const statusText = document.getElementById("statusText")
const accuracyText = document.getElementById("accuracyText")
const targetColumnText = document.getElementById("targetColumnText")
const refreshStatusBtn = document.getElementById("refreshStatusBtn")
const predictionSection = document.getElementById("predictionSection")
const predictionForm = document.getElementById("predictionForm")
const featureInputsDiv = document.getElementById("featureInputs")
const predictionResultDiv = document.getElementById("predictionResult")

let currentModelFeatures = null // ישמור את הפיצ'רים שהמודל אומן עליהם

// פונקציית עזר להצגת הודעות
function displayMessage(element, message, type) {
  element.textContent = message
  element.className = `message ${type}`
  element.style.display = "block"
}

// פונקציה להעלאת קובץ CSV ואימון המודל
async function uploadAndTrainModel(event) {
  event.preventDefault()
  uploadStatus.style.display = "none" // הסתר הודעות קודמות

  const file = csvFile.files[0]
  if (!file) {
    displayMessage(uploadStatus, "אנא בחר קובץ CSV.", "error")
    return
  }

  const formData = new FormData()
  formData.append("file", file)

  try {
    const response = await fetch(`${BACKEND_URL}/train`, {
      method: "POST",
      body: formData,
    })

    const data = await response.json()

    if (response.ok) {
      displayMessage(uploadStatus, `מודל אומן בהצלחה! דיוק: ${data.accuracy.toFixed(2)}%`, "success")
      // רענן את סטטוס המודל לאחר אימון מוצלח
      await fetchModelStatus()
    } else {
      displayMessage(uploadStatus, `שגיאה באימון המודל: ${data.detail || "נסה שוב."}`, "error")
    }
  } catch (error) {
    console.error("שגיאה ברשת או בשרת:", error)
    displayMessage(uploadStatus, `שגיאה בחיבור לשרת: ${error.message}`, "error")
  }
}

// פונקציה לקבלת סטטוס המודל מה-Backend
async function fetchModelStatus() {
  try {
    const response = await fetch(`${BACKEND_URL}/status`)
    const data = await response.json()

    if (response.ok) {
      statusText.textContent = data.status === "Model trained" ? "מודל אומן" : "אין מודל מאומן"
      accuracyText.textContent = data.accuracy !== null ? `${data.accuracy.toFixed(2)}%` : "--"
      targetColumnText.textContent = data.target_column || "--"

      if (data.status === "Model trained" && data.features) {
        currentModelFeatures = data.features
        createPredictionForm(data.features)
        predictionSection.style.display = "block" // הצג את סעיף החיזוי
      } else {
        currentModelFeatures = null
        predictionSection.style.display = "none" // הסתר את סעיף החיזוי
        featureInputsDiv.innerHTML = "" // נקה את הטופס
      }
    } else {
      displayMessage(modelStatusDiv, `שגיאה בקבלת סטטוס: ${data.detail || "נסה שוב."}`, "error")
    }
  } catch (error) {
    console.error("שגיאה ברשת או בשרת:", error)
    displayMessage(modelStatusDiv, `שגיאה בחיבור לשרת: ${error.message}`, "error")
  }
}

// פונקציה ליצירת טופס החיזוי באופן דינמי
function createPredictionForm(features) {
  featureInputsDiv.innerHTML = "" // נקה אלמנטים קודמים

  for (const featureName in features) {
    const values = features[featureName]

    const groupDiv = document.createElement("div")
    groupDiv.className = "feature-input-group"

    const label = document.createElement("label")
    label.setAttribute("for", `feature-${featureName}`)
    label.textContent = featureName
    groupDiv.appendChild(label)

    if (values.length > 0) {
      // אם יש ערכים מוגדרים, צור סלקט (dropdown)
      const select = document.createElement("select")
      select.id = `feature-${featureName}`
      select.name = featureName
      select.required = true

      // הוסף אופציה ריקה כברירת מחדל
      const defaultOption = document.createElement("option")
      defaultOption.value = ""
      defaultOption.textContent = `בחר ${featureName}...`
      defaultOption.disabled = true
      defaultOption.selected = true
      select.appendChild(defaultOption)

      values.forEach((value) => {
        const option = document.createElement("option")
        option.value = value
        option.textContent = value
        select.appendChild(option)
      })
      groupDiv.appendChild(select)
    } else {
      // אם אין ערכים מוגדרים (לדוגמה, פיצ'ר מספרי או טקסט חופשי), צור שדה טקסט
      const input = document.createElement("input")
      input.type = "text"
      input.id = `feature-${featureName}`
      input.name = featureName
      input.placeholder = `הכנס ערך עבור ${featureName}`
      input.required = true
      groupDiv.appendChild(input)
    }
    featureInputsDiv.appendChild(groupDiv)
  }
}

// פונקציה לשליחת בקשת חיזוי
async function submitPrediction(event) {
  event.preventDefault()
  predictionResultDiv.style.display = "none" // הסתר הודעות קודמות

  if (!currentModelFeatures) {
    displayMessage(predictionResultDiv, "אין מודל מאומן. אנא אמן מודל תחילה.", "error")
    return
  }

  const features = {}
  let allInputsValid = true

  for (const featureName in currentModelFeatures) {
    const inputElement = document.getElementById(`feature-${featureName}`)
    if (inputElement) {
      const value = inputElement.value.trim()
      if (value === "") {
        allInputsValid = false
        break
      }
      features[featureName] = value
    } else {
      allInputsValid = false // Input element not found for a feature
      break
    }
  }

  if (!allInputsValid) {
    displayMessage(predictionResultDiv, "אנא מלא את כל השדות.", "error")
    return
  }

  try {
    const response = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ features }),
    })

    const data = await response.json()

    if (response.ok) {
      let resultHtml = `<h3>החיזוי: ${data.prediction}</h3>`
      resultHtml += "<p>הסתברויות לכל קלאס:</p><ul>"
      for (const [className, probability] of Object.entries(data.full_results)) {
        resultHtml += `<li>${className}: ${(probability * 100).toFixed(2)}%</li>`
      }
      resultHtml += "</ul>"
      predictionResultDiv.innerHTML = resultHtml
      predictionResultDiv.className = "message success"
      predictionResultDiv.style.display = "block"
    } else {
      displayMessage(predictionResultDiv, `שגיאה בחיזוי: ${data.detail || "נסה שוב."}`, "error")
    }
  } catch (error) {
    console.error("שגיאה ברשת או בשרת:", error)
    displayMessage(predictionResultDiv, `שגיאה בחיבור לשרת: ${error.message}`, "error")
  }
}

// הוספת מאזיני אירועים
uploadForm.addEventListener("submit", uploadAndTrainModel)
refreshStatusBtn.addEventListener("click", fetchModelStatus)
predictionForm.addEventListener("submit", submitPrediction)

// טען סטטוס מודל בעת טעינת הדף
document.addEventListener("DOMContentLoaded", fetchModelStatus)
