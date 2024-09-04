# (German) Forest Fire Prediction Model

# What?

This is my first true independent machine learning project - so don't expect too much!

This is a compilation of models to predict the damage of forest fires in Germany.

After initial exploration and data pre-processing, this notebook comprises two models: 

- A supervised learning model to predict the average temperature in the selected region of Germany in future years.
- A supervised learning model to predict the total area of forest that will be burnt based on that year's average temperature.

I feed the predicted future temperatures from the first model into the second to get predictions of the area that will be burn over coming years.

# Positive Reflections

- It works:) As my first independent project, this was the priority.
- The notebook's inline visualisations work nicely to allow the reader to see how future values compare to historical trends
- This is a basis to expand and improve these skills, in the hope that I'll one day draw useful conclusions about something I care about!
  
# For the future
- The conclusions aren't useful; due to sparse, generalised data (the average temperature across all seasons was used) leaves little scope for useful conclusions to be drawn
- Some years there were no fires, others there were some. A more sophisticated model could include the historical likelihood of there being no fire at all in a given year.
- With an eye on prevention, it would be useful to integrate the cause of fires, or reverse engineer insights about what caused a fire based on the fires characteristics :)
