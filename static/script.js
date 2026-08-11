const imageInput = document.getElementById("imageInput");
const uploadText = document.getElementById("uploadText");
const previewWrapper = document.getElementById("previewWrapper");
const previewImage = document.getElementById("previewImage");
const predictBtn = document.getElementById("predictBtn");
const spinner = document.getElementById("spinner");
const statusText = document.getElementById("statusText");
const resultCard = document.getElementById("resultCard");

const diseaseName = document.getElementById("diseaseName");
const confidenceScore = document.getElementById("confidenceScore");
const diseaseDescription = document.getElementById("diseaseDescription");
const symptomsList = document.getElementById("symptomsList");
const precautionsList = document.getElementById("precautionsList");
const treatmentList = document.getElementById("treatmentList");

let selectedFile = null;

imageInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;

  selectedFile = file;
  uploadText.textContent = file.name;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewWrapper.classList.remove("hidden");
  };
  reader.readAsDataURL(file);

  predictBtn.disabled = false;
  resultCard.classList.add("hidden");
  statusText.textContent = "";
});

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  setLoading(true);
  resultCard.classList.add("hidden");
  statusText.textContent = "";

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    renderResult(data);
  } catch (err) {
    statusText.textContent = `Error: ${err.message}`;
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  predictBtn.disabled = isLoading;
  spinner.classList.toggle("hidden", !isLoading);
  statusText.textContent = isLoading ? "Analyzing image..." : "";
}

function fillList(listElement, items) {
  listElement.innerHTML = "";
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.textContent = "Not available";
    listElement.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    listElement.appendChild(li);
  });
}

function renderResult(data) {
  diseaseName.textContent = data.predicted_class;
  confidenceScore.textContent = `${data.confidence}%`;
  diseaseDescription.textContent = data.description;

  fillList(symptomsList, data.symptoms);
  fillList(precautionsList, data.precautions);
  fillList(treatmentList, data.treatment);

  resultCard.classList.remove("hidden");
}
