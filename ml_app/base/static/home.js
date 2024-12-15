function validateForm() {
    const form = document.getElementById('sleepForm');
    const inputs = form.querySelectorAll('input, select');
    let isValid = true;

    inputs.forEach((input) => {
        const errorSpan = input.nextElementSibling;
        if (input.value.trim() === '') {
            isValid = false;
            if (errorSpan && errorSpan.classList.contains('error-message')) {
                errorSpan.style.display = 'block';
                errorSpan.textContent = 'This field is required';
            } else {
                const error = document.createElement('span');
                error.className = 'error-message text-danger';
                error.textContent = 'This field is required';
                error.style.display = 'block';
                input.insertAdjacentElement('afterend', error);
            }
        } else {
            if (errorSpan && errorSpan.classList.contains('error-message')) {
                errorSpan.style.display = 'none';
            }
        }
    });

    return isValid;
}

async function logFormValues() {
    if (validateForm()) {
        const form = document.getElementById('sleepForm');
        const formData = new FormData(form);
        const values = Object.fromEntries(formData.entries());

        console.log('Form values:', values);
        await submit_info(values);
    } else {
        console.log('Form validation failed.');
    }
}

async function submit_info(data) {
    const url = '/process-data/';
    const csrfToken = getCookie('csrftoken');
    if (!csrfToken) {
        console.error('CSRF token not found.');
        return;
    }

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            },
            body: JSON.stringify({ data }),
        });

        const responseData = await response.json();

        if (response.ok) {
            console.log('Success:', responseData);
            const prediction = responseData.prediction;

            // Update the UI with the prediction
            const predictionElement = document.getElementById('prediction-display');
            if (predictionElement) {
                predictionElement.textContent = `Predicted Sleep Quality: ${prediction}`;
                predictionElement.style.display = 'block'; // Make sure it's visible
            }
            // Process response data (e.g., show prediction)
        } else {
            console.error('Error:', responseData);
        }
    } catch (error) {
        console.error('Fetch error:', error);
    }
}

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.startsWith(name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// Add event listener to the form submit button
// document.addEventListener('DOMContentLoaded', function() {
//     const submitButton = document.querySelector('button[type="button"]');
//     if (submitButton) {
//         submitButton.addEventListener('click', logFormValues);
//     }
// });

