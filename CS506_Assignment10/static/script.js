document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("#search-form");
    const queryTypeDropdown = document.querySelector("#query-type");
    const numComponentsContainer = document.querySelector("#num-components-container");
    queryTypeDropdown.addEventListener("change", () => {
        if (queryTypeDropdown.value === "image") {
            numComponentsContainer.style.display = "block"; // Show the field
        } else {
            numComponentsContainer.style.display = "none"; // Hide the field
        }
    });
    const clearButton = document.getElementById("clear-file");
    const fileInput = document.getElementById("image-query");

    clearButton.addEventListener("click", () => {
        fileInput.value = ""; // Clear the selected file
    });

    form.addEventListener("submit", async (event) => {
            event.preventDefault(); // Prevent the default form submission
            const fileInput = document.getElementById('image-query');
            const formData = new FormData();
            const queryType = document.querySelector("#query-type").value;
            const imageQuery = fileInput.files[0];
            const textQuery = document.querySelector("#text-query").value;
            const hybridWeight = document.querySelector("#hybrid-weight").value;
            const numComponents=  document.querySelector("#num-components").value;

            // Prepare JSON payload
            formData.append("query-type",queryType)
            if (imageQuery && (queryType === "hybrid" || queryType === "image")) {
                formData.append("image-query", imageQuery);
            }
            if (textQuery && (queryType === "hybrid" || queryType === "text")) {
                formData.append("text-query", textQuery);
            }
            if (hybridWeight) {
               formData.append("hybrid-query", hybridWeight);
            }
            if(numComponents&&(queryType==='image')){
                formData.append('numComponents',numComponents)
            }
            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    body:formData,
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }

                const result = await response.json();
                if (result.status === "success") {
                    displayResults(result.results);
                } else {
                    alert("Error: " + result.message);
                }
            } catch (error) {
                console.error("Error submitting form:", error);
                alert("An unexpected error occurred. Please try again.");
            }
        });

        function displayResults(results) {
            const resultsContainer = document.querySelector(".results");
            resultsContainer.innerHTML = "<h2>Top Search Results</h2>";

            results.forEach((result) => {
                const resultItem = document.createElement("div");
                resultItem.classList.add("result-item");

                const img = document.createElement("img");
                img.src = result.file_name; // Use file_name as the image URL
                img.alt = "Result";

                const span = document.createElement("span");
               if ("similarity" in result) {
                span.textContent = `Cosine Similarity: ${result.similarity.toFixed(3)}`;
            } else if ("distances" in result) {
                span.textContent = `Distance: ${result.distances.toFixed(3)}`;
            } else {
                span.textContent = "No similarity or distance data available.";
            }

                resultItem.appendChild(img);
                resultItem.appendChild(span);
                resultsContainer.appendChild(resultItem);
            });
        }
    });