from flask import Flask, request, jsonify
import pyodbc  # Use the appropriate SQL Server driver
import pandas as pd
import numpy as np
import joblib as jbl

app = Flask(__name__)

# ------------- ORIGINAL DATASET -------------------------------------------------------------

etl_data = pd.read_csv(r'etl_df.csv')


# CALCULATE MEAN & STANDARD DEVIATION OF ORIGINAL DATASET
def column_mean_std(dataset):
    means, std = list(), list()

    for col in dataset.loc[:, dataset.columns != 'NUM']:
        means.append(round(dataset[col].mean(), 3))
        std.append(round(dataset[col].std(), 3))

    return means, std


# COLLECT MEAN & STANDARD DEVIATION OF ORIGINAL DATASET
means, std = column_mean_std(etl_data)


# --------------- PREPROCESSING USER INPUT -----------------------------------------------------
# STANDARDIZATION OF INPUT DATA

def standardize_data(input_data, means, std):
    scaled_data = []

    for i in range(len(input_data)):
        scaled_data.append((input_data[i] - means[i]) / std[i])

    return scaled_data


# ------------- CONNECTION TO MS SQL DATABASE ---------------------------------------------------------------

server_name = r'VICKYS\SQLEXPRESS'
database = 'AVLDatabase'
connectionString = 'Driver= {SQL Server};Server=' + server_name + ';Database = ' + database + ';Trusted_connection=yes'
connection = pyodbc.connect(connectionString)

@app.route('/database_fetch', methods=['GET'])
def database_fetch():
    print('Fetching data from Database')

    try:
        # Establish a connection to the SQL Server database
        # connection = pyodbc.connect(connectionString)
        cursor = connection.cursor()
        query = 'SELECT top 10 * FROM [AVLDatabase].[dbo].[tbl_fleetdata]'
        cursor.execute(query)

        # Fetch the query result
        data = cursor.fetchall()

        # Converting the result to a dictionary for serialization to JSON
        data_json = [dict(zip([column[0] for column in cursor.description], row)) for row in data]

        # Close the connection
        connection.close()

        return jsonify(data_json)

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/<int:vehicle_id>', methods=['DELETE'])
def delete_data(vehicle_id):
    try:
        cursor = connection.cursor()
        query = f"DELETE FROM [AVLDatabase].[dbo].[tbl_vehicledata] WHERE PSEUDO_VIN = ?"
        cursor.execute(query, (vehicle_id,))
        connection.commit()
        cursor.close()
        return jsonify({"message": f"Vehicle with PSEUDO_VIN {vehicle_id} deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        connection.close()


@app.route('/failure_pred_api', methods=['POST'])
def failure_pred_api():
    # REQUEST RECEIVED FROM API
    api_input = request.get_json()
    scaling_input = [sublist for mainlist in api_input for sublist in mainlist]

    print('Request from Jupyter Notebook ')
    print(scaling_input)

    # STANDARDIZATION OF INPUT DATA & RESCALING INTO 1-BY-22 DIMENSIONAL ARRAY
    scaled_input = standardize_data(scaling_input, means, std)
    scaled_input = np.array(scaled_input).reshape(-1, 12)
    print('SCALED API INPUT')
    print((scaled_input))

    # PREDICTIONS
    prediction_output = model_output.predict_proba(scaled_input)
    prediction_to_api_request = np.array2string(model_output.predict_proba(scaled_input))
    # prediction_to_api_request = np.array2string(model_output.predict(scaled_input))

    # COLLECTING EACH FEATURE & COMBINING WITH OUTPUT
    # combined_data = collect_data(scaling_input, prediction_output)

    # UPDATE COLLECTED DATA TO THE DATABASE
    # try:
    #     update_database_postgre(combined_data, myconnect_postgre)
    #     print('......')
    #     print('..........')
    #     print('................')
    #     print('..........................')
    #     update_database(combined_data, myconnect)
    # except:
    #     print('CONNECTION TO DATABASE FAILED')

    # RETURNING THE PREDICTIONS TO API CALL

    return jsonify(prediction_to_api_request)


if __name__ == '__main__':
    # LOAD MODEL
    load_model = 'failure_prediction.sav'
    model_output = jbl.load(load_model)

    # RUN THE APP
    # app.run(host='0.0.0.0', port=8080)
    app.run(debug=True)
