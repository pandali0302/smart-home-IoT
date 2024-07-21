<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSE</th>
      <th>MAE</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Baseline(MA)</th>
      <td>0.053</td>
      <td>0.187</td>
    </tr>
    <tr>
      <th>Arima</th>
      <td>0.055</td>
      <td>0.186</td>
    </tr>
    <tr>
      <th>LightGBM</th>
      <td>0.049</td>
      <td>0.171</td>
    </tr>
    <tr>
      <th>LSTM_Multivar</th>
      <td>0.020</td>
      <td>0.114</td>
    </tr>
  </tbody>
</table>
</div>