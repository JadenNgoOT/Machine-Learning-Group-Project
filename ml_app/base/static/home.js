function validateForm() {
    const form = document.getElementById('sleepForm');
    const inputs = form.querySelectorAll('input, select, textarea'); // Include all input types
    let isValid = true;

    inputs.forEach((input) => {
        const errorSpan = input.nextElementSibling; // Assumes error span is next to the input
        if (input.value.trim() === '') {
            // Add error if missing
            if (!errorSpan || !errorSpan.classList.contains('error-message')) {
                const error = document.createElement('span');
                error.className = 'error-message text-danger';
                error.textContent = 'This field is required';
                input.insertAdjacentElement('afterend', error);
            }
            isValid = false;
        } else {
            // Remove existing error
            if (errorSpan && errorSpan.classList.contains('error-message')) {
                errorSpan.remove();
            }
        }
    });

    return isValid;
}

async function logFormValues() {
    if (!validateForm()) {
        console.log('Form validation failed.');
        return; // Stop processing if the form is invalid
    }

    const form = document.getElementById('sleepForm');
    const formData = new FormData(form);
    const values = {};

    formData.forEach((value, key) => {
        values[key] = value;
    });

    console.log('Form values:', values); // Debugging output
    await submit_info(values);
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
            // Process response data (e.g., show prediction)
        } else {
            console.error('Error:', responseData);
        }
    } catch (error) {
        console.error('Fetch error:', error);
    }
}

// Helper function to get CSRF token
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
