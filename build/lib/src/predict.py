def predict(model, data, target_names):
    prediction = model.predict([data])
    return target_names[prediction[0]]
