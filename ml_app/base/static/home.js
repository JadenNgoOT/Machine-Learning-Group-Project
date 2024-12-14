async function submit_info(data){
    url = "/process-data/";
    try {
        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type' : 'application/json',
                'X-CSRFToken' : getCookie('csrftoken'),
            },
            body: JSON.stringify({
                'data' : data,
            })
        }).then(response => response.json())
        .then(data => {
            if (data.status === '200'){
                // debug message
                console.log("Success");
                console.log(data.prediction)
                // handle return from api view
            } else {
                console.error("Error");
            }
        })
    } catch (error) {
        console.error("Error: ", error);
    }
}

function logFormValues() {
    const form = document.getElementById('sleepForm');
    const formData = new FormData(form);
    const values = {};
    formData.forEach((value, key) => {
        values[key] = value;
    });
    // console.log(values);
    submit_info(values);
}

// Helper function needed to make API calls.
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === name + '=') {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}