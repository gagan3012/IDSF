models=(mt0-small mt0-base mt0-large)
datasets=(setting5 setting6 setting7 setting8 setting9)
columns=(intents)

for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for column in "${columns[@]}"
        do
            job_name="$model-$dataset-$column"
            echo $job_name
            sbatch --job-name=$job_name batch_train_v3.sh $model $column $dataset
        done
    done
done