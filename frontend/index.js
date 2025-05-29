

document.getElementById("form").addEventListener("submit", async function (event) {
    event.preventDefault();
    console.log("Submitted");
    const form = document.getElementById("form");
    const formData = new FormData(form);



    const prediction = await getPrediction(formData);
    console.log(prediction);

    document.getElementById("output").textContent = "The bird species is " + prediction + ". "

});



async function getPrediction(formData) {
    const response = await fetch("http://127.0.0.1:5001/identify", {
        method: "POST",
        body: formData,

    });

    const data = await response.json();
    return data.result;

}
