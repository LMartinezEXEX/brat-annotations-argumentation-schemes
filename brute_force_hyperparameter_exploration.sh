for batch_size in 16 32 64
    do
    for lr in 1e-05 2e-05 5e-05 5e-04 1e-04 1e-03 5e-06
        do
        for modelname in roberta-base distilbert-base-uncased distilroberta-base pysentimiento/robertuito-base-uncased
        do
            python3 train_model_for_component.py pivot --modelname ${modelname} --lr ${lr} --batch_size ${batch_size}
        done
    done
done
