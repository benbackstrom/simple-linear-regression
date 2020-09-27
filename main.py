
import pandas
import trainer
import plotter
import hyperparameters


# Get hyperparameters for each column
hyper_params = hyperparameters.get()
label_name = 'Total'


# Create DataFrame of input
df = pandas.read_csv('deaths_spokane.csv')\
    .astype(int)\
    .transpose()\
    .rename(columns={0: hyper_params[0]['name'],
                     1: hyper_params[1]['name'],
                     2: hyper_params[2]['name'],
                     3: hyper_params[3]['name'],
                     4: hyper_params[4]['name']})

print(df)
print()


# Loop through features and train with given hyperparameters
for params in hyper_params:
    column_name = params['name']
    if column_name == 'Cancer' or column_name == 'Heart Disease' or column_name == 'Alsheimers' or column_name == 'Total':
        continue

    weight, bias, error, epoch_data = trainer.train_model(feature=df[column_name],
                                                          label=df[label_name],
                                                          learning_rate=params['learning_rate'],
                                                          number_epochs=params['epochs'],
                                                          batch_size=params['batch_size'])
    print(f'bias={bias}, weight={weight}')
    print()

    plotter.plot_model(title='Causes of Death',
                       feature_title=column_name,
                       label_title=label_name,
                       weight=weight,
                       bias=bias,
                       feature_data=df[column_name],
                       label_data=df[label_name])
    plotter.plot_loss(epoch_data=epoch_data,
                      root_mean_squared_error=error)
