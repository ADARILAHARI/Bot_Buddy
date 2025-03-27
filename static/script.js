function sendMessage() {
    let userInput = document.getElementById("user-input").value;
    document.getElementById("user-input").value = "";

    let chatbox = document.getElementById("chatbox");
    chatbox.innerHTML += "<p><b>You:</b> " + userInput + "</p>";

    fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        chatbox.innerHTML += "<p><b>Bot:</b> " + data.response + "</p>";
        chatbox.scrollTop = chatbox.scrollHeight;
    })
    .catch(error => console.error("Error:", error));
}