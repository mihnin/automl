document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('forecast-form');
    const csvFileInput = document.getElementById('csv-file');
    const targetColumnSelect = document.getElementById('target-column');
    const dateColumnSelect = document.getElementById('date-column');
    const categoryColumnSelect = document.getElementById('category-column');
    const resultsDiv = document.getElementById('results');

    csvFileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        const reader = new FileReader();

        reader.onload = function(e) {
            const contents = e.target.result;
            const lines = contents.split('\n');
            const headers = lines[0].split(',');

            // Очистка существующих опций
            [targetColumnSelect, dateColumnSelect, categoryColumnSelect].forEach(select => {
                select.innerHTML = '';
            });

            // Добавление новых опций
            headers.forEach(header => {
                const option = document.createElement('option');
                option.value = header.trim();
                option.textContent = header.trim();

                targetColumnSelect.appendChild(option.cloneNode(true));
                dateColumnSelect.appendChild(option.cloneNode(true));
                categoryColumnSelect.appendChild(option.cloneNode(true));
            });

            // Добавление пустой опции для категориального признака
            const emptyOption = document.createElement('option');
            emptyOption.value = '';
            emptyOption.textContent = 'Не выбрано';
            categoryColumnSelect.insertBefore(emptyOption, categoryColumnSelect.firstChild);
            categoryColumnSelect.value = '';
        };

        reader.readAsText(file);
    });

    form.addEventListener('submit', function(event) {
        event.preventDefault();

        const formData = new FormData(form);

        fetch('/forecast', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Ошибка:', error);
            alert('Произошла ошибка при выполнении прогноза. Пожалуйста, попробуйте еще раз.');
        });
    });

    function displayResults(data) {
        resultsDiv.style.display = 'block';

        // Отображение графика
        Plotly.newPlot('chart', [{
            x: data.dates,
            y: data.actual,
            type: 'scatter',
            mode: 'lines',
            name: 'Фактические данные'
        }, {
            x: data.dates,
            y: data.predicted,
            type: 'scatter',
            mode: 'lines',
            name: 'Прогноз'
        }], {
            title: 'Фактические данные и прогноз',
            xaxis: { title: 'Дата' },
            yaxis: { title: 'Значение' }
        });

        // Отображение таблицы с прогнозом
        const table = document.getElementById('forecast-table');
        table.innerHTML = `
            <tr>
                <th>Дата</th>
                <th>Фактическое значение</th>
                <th>Прогноз</th>
            </tr>
        `;

        for (let i = 0; i < data.dates.length; i++) {
            const row = table.insertRow();
            row.insertCell(0).textContent = data.dates[i];
            row.insertCell(1).textContent = data.actual[i] || '-';
            row.insertCell(2).textContent = data.predicted[i];
        }
    }
});
