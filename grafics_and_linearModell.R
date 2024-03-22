install.packages("lubridate")
install.packages("ggplot2")
install.packages("tidyr")
install.packages("corrplot")
install.packages("forecast")


#library(readr)
#library(ggplot2)
#library(tidyr)
#library(dplyr)
#library(lubridate)
library(tidyverse)
library(corrplot)



df <- read_csv("C:/Programming/hslu_masterthesis/Masterthesis/notebooks/240226_Combined_Load_And_TrafoData.csv")

df
df$year <- as.factor(df$year)
df$quarter <- as.factor(df$quarter)
df$month <- as.factor(df$month)
df$day <- as.factor(df$day)
df$hour <- as.factor(df$hour)
df$day_of_week <- as.factor(df$day_of_week)
df$is_weekend <- as.factor(df$is_weekend)
df$datetime <- as.POSIXct(df$datetime)


df

# Plot trafo_p_lv_mw, trafo_loading_percent, trafo_q_lv_mvar

# Load necessary libraries
library(ggplot2)
library(tidyr)

# Assuming df has an implicit time order, if there's an explicit time column, use that instead

# Add a time index to the dataframe
df$time_index <- 1:nrow(df)

# Convert df from wide to long format for easier plotting with ggplot
df_long <- tidyr::pivot_longer(df, cols = c(trafo_p_lv_mw, trafo_loading_percent, trafo_q_lv_mvar), 
                               names_to = "variable", values_to = "value")

# Plotting
ggplot(df_long, aes(x = time_index, y = value, color = variable)) + 
  geom_line() + 
  labs(x = "Time", y = "Value", title = "Time Series Plot") +
  theme_minimal() +
  scale_color_manual(values = c("trafo_p_lv_mw" = "blue", 
                                "trafo_loading_percent" = "red", 
                                "trafo_q_lv_mvar" = "green"))


# Remove columns which are not used
df$'...1' <- NULL
df$trafo_q_hv_mvar <- NULL
df$trafo_p_hv_mw <- NULL
df$trafo_q_lv_mvar <- NULL
df$trafo_pl_mw <- NULL
df$trafo_va_lv_degree <- NULL
df$trafo_vm_hv_pu <- NULL
df$trafo_va_hv_degree <- NULL
df$trafo_ql_mvar <- NULL
df$trafo_i_hv_ka <- NULL
df$trafo_i_lv_ka <- NULL
df$trafo_loading_percent <- NULL
df$trafo_vm_lv_pu <- NULL
df$year <- NULL

df

# Decomposition
trafo_ts <- ts(df$trafo_p_lv_mw, frequency = 4)  # Change 'frequency' based on your actual data time intervals

# Perform decomposition
# For STL decomposition
library(forecast)
decomp_result <- stl(trafo_ts, s.window = "periodic")

# For classical decomposition
# decomp_result <- decompose(trafo_ts)

# Plot the decomposed components
plot(decomp_result)





df_long <- pivot_longer(df, cols = starts_with("load_"), 
                        names_to = "load_type", values_to = "load_value")

# Create a new column for date only (without time)
df_long$date <- as.Date(df_long$datetime)

# Calculate daily averages for each load type
daily_load_averages <- df_long %>%
  group_by(date, load_type) %>%
  summarize(daily_avg = mean(load_value, na.rm = TRUE))

# Now plot the smoothed curves for each load type over time
ggplot(data = daily_load_averages, aes(x = date, y = daily_avg, colour = load_type)) +
  geom_line() +  # adds the line connecting daily averages
  labs(title = "Smoothed Daily Load Curves",
       x = "Date",
       y = "Average Load Value",
       colour = "Load Type") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Correlation plot
load_data <- select(df, starts_with("load_"))
correlation_matrix <- cor(load_data, use = "complete.obs")  # 'use' parameter handles missing values
corrplot(correlation_matrix, method = "circle", order = "hclust",
         tl.col = "black", tl.srt = 45)  # Adjust parameters as needed

corrplot(correlation_matrix, method = "number", order = "hclust",
         tl.col = "black", tl.srt = 45)


# Linear Modell
lm1 <- lm(trafo_p_lv_mw ~ ., data = df)

summary(lm1)

plot(lm1)



# Addressing Colinearity
library(car)
library(MASS)  # For stepwise selection



# Prepare matrix of predictors and response vector
x_matrix <- model.matrix(trafo_p_lv_mw ~ ., data = df)[,-1]  # Remove intercept column
y_vector <- df$trafo_p_lv_mw

# Lasso regression
lasso_model <- glmnet(x_matrix, y_vector, alpha = 1)
summary(lasso_model)
plot(lasso_model)

# Ridge regression
ridge_model <- glmnet(x_matrix, y_vector, alpha = 0)
plot(ridge_model)