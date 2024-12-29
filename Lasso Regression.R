# LOAD PACKAGES
pacman::p_load(tidyverse, vip, tidymodels, ggplot2)

# GET DATA
ratings_raw <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-03-17/office_ratings.csv")

# SPECIFY A REGEX TO REMOVE PUNCTUATION, NUMBERS, AND COMMON WORDS
remove_regex <- "[:punct:]|[:digit:]|parts |part |the |and"

# FORMAT THE DATA
office_ratings <- ratings_raw %>%
  transmute(
    episode_name = str_to_lower(title),
    episode_name = str_remove_all(episode_name, remove_regex),
    episode_name = str_trim(episode_name),
    imdb_rating
  )

# LOAD AND PREPROCESS EPISODE INFORMATION
office_info <- schrute::theoffice %>%
  mutate(
    season = as.numeric(season),
    episode = as.numeric(episode),
    episode_name = str_to_lower(episode_name),
    episode_name = str_remove_all(episode_name, remove_regex),
    episode_name = str_trim(episode_name)
  ) %>%
  select(season, episode, episode_name, director, writer, character)

office_info

# CREATE A DATA FRAME OF CHARACTER APPEARANCES PER EPISODE
characters <- office_info %>%
  # CREATE A COUNT OF HOW MANY TIME EACH CHARACTER APPEARS IN AN EPISODE
  count(episode_name, character) %>%
  # ADD A COUNT OF HOW MANY TIME EACH CHARACTER APPEARS ACROSS ALL EPISODES
  add_count(character, wt = n, name = "character_count") %>%
  filter(character_count > 800) %>%
  select(-character_count) %>%
  pivot_wider(
    names_from = character,
    values_from = n,
    values_fill = list(n = 0)
  )

characters

creators <- office_info %>%
  distinct(episode_name, director, writer) %>%
  pivot_longer(director:writer, names_to = "role", values_to = "person") %>%
  separate_rows(person, sep = ";") %>%
  add_count(person) %>%
  filter(n > 10) %>%
  distinct(episode_name, person) %>%
  mutate(person_value = 1) %>%
  pivot_wider(
    names_from = person,
    values_from = person_value,
    values_fill = list(person_value = 0)
  )

creators

# CREATE AN OFFICE DATASET
office <- office_info %>%
  distinct(season, episode, episode_name) %>%
  inner_join(characters, by = "episode_name") %>%
  inner_join(creators, by = "episode_name") %>%
  inner_join(office_ratings %>%
    select(episode_name, imdb_rating) %>%
    group_by(episode_name) %>%
    summarise(imdb_rating = mean(imdb_rating), .groups = "drop"), by = "episode_name") %>%
  janitor::clean_names()

office

# CREATE A BOXPLOT OF RATINGS BY EPISODE NUMBER
office %>%
  ggplot(aes(episode, imdb_rating, fill = as.factor(episode))) +
  geom_boxplot(show.legend = FALSE) +
  labs(title = "IMDB Ratings by Episode", x = "Episode", y = "IMDB Rating") +
  theme(plot.title = element_text(hjust = 0.5))


# TRAIN THE MODEL ---------------------------------------------------------

# SAVE THE SPLIT INFORMATION FOR AN 80/20 SPLIT OF THE DATA
office_split <- initial_split(office, strata = season, prop = 0.8)

# CREATE A TRAINING SPLIT
office_train <- training(office_split)

# CREATE A TEST SPLIT
office_test <- testing(office_split)

# CREATE A PREPROCESSING RECIPE
office_rec <- recipe(imdb_rating ~ ., data = office_train) %>%
  update_role(episode_name, new_role = "ID") %>%
  step_zv(all_numeric(), -all_outcomes()) %>%
  step_normalize(all_numeric(), -all_outcomes())

# PRE-PROCESS THE DATA
office_prep <- office_rec %>%
  prep(strings_as_factors = FALSE) %>%
  bake(new_data = NULL)

# DEFINE A LASSO REGRESSION MODEL
lasso_spec <- linear_reg(penalty = 0.1, mixture = 1) %>%
  set_engine("glmnet")

# CREATE A WORKFLOW
wf <- workflow() %>%
  add_recipe(office_rec)

# TRAIN THE LASSO REGRESSION MODEL
lasso_fit <- wf %>%
  add_model(lasso_spec) %>%
  fit(data = office_train)

# TUNE THE LASSO PARAMETERS -----------------------------------------------

set.seed(345)
# CREATE 10 FOLD CROSS VALIDATION RESAMPLES
folds <- vfold_cv(office_train, v = 10)

reg_lasso_spec <- lasso_spec

# DEFINE A TUNABLE LASSO REGRESSION MODEL SPECIFICATION
lm_lasso_spec_tune <-
  linear_reg() %>%
  set_args(mixture = tune(), penalty = tune()) %>%
  set_engine(engine = "glmnet", path_values = pen_vals) %>%
  set_mode("regression")

# CREATE A WORKFLOW FOR TUNING THE LASSO REGRESSION MODEL
lasso_wf_tune <- workflow() %>%
  add_recipe(office_rec) %>%
  add_model(lm_lasso_spec_tune)

# DEFINE A SEQUENCE OF PENALTY VALUES FOR TUNING
pen_vals <- 10^seq(-3, 0, length.out = 10)

# CREATE A GRID OF HYPER-PARAMETER VALUES FOR TUNING
grid <- crossing(penalty = pen_vals, mixture = c(0.1, 1.0))

# DEFINE A FUNCTION TO EXTRACT COEFFICIENTS FROM A GLMNET MODEL
get_glmnet_coefs <- function(x) {
  x %>%
    extract_fit_engine() %>%
    tidy(return_zeros = TRUE) %>%
    rename(penalty = lambda)
}

# SET UP CONTROL PARAMETERS FOR TUNING WITH COEFFICIENT EXTRACTION
parsnip_ctrl <- control_grid(extract = get_glmnet_coefs)

# TUNE THE LASSO REGRESSION MODEL USING A GRID SEARCH
glmnet_res <- lasso_wf_tune %>%
  tune_grid(
    resamples = folds,
    grid = grid,
    metrics = metric_set(rmse, mae),
    control = parsnip_ctrl
  )

glmnet_res

glmnet_res %>%
  collect_metrics() %>%
  ggplot(aes(penalty, mean, colour = .metric)) +
  geom_errorbar(
    aes(
      ymin = mean - std_err,
      ymax = mean + std_err
    ),
    alpha = 0.5
  ) +
  geom_line(size = 1.5, show.legend = FALSE) +
  facet_wrap(~.metric, scales = "free", nrow = 2) +
  scale_x_log10() +
  theme(legend.position = 'none')

# EXTRACT AND TIDY THE COEFFICIENTS FROM THE TUNING RESULTS
glmnet_coefs <- glmnet_res %>%
  select(id, .extracts) %>%
  unnest(.extracts) %>%
  select(id, mixture, .extracts) %>%
  group_by(id, mixture) %>%
  slice(1) %>%
  ungroup() %>%
  unnest(.extracts)

# VISUALIZE THE LASSO COEFFICIENTS ACROSS DIFFERENT PENALTY VALUES AND MIXTURE LEVELS
glmnet_coefs %>%
  filter(term != "(Intercept)") %>%
  mutate(mixture = format(mixture)) %>%
  ggplot(aes(x = penalty, y = estimate, col = mixture, groups = id)) +
  geom_hline(yintercept = 0, lty = 3) +
  geom_line(alpha = 0.5, lwd = 1.2) +
  facet_wrap(~term) +
  scale_x_log10() +
  scale_color_brewer(palette = "Accent") +
  labs(y = "coefficient") +
  theme(legend.position = "top")

# VISUALISE THE MODEL EVALUATION METRICS
autoplot(glmnet_res) +
  theme_classic()

# SUMMARISE THE MODEL EVALUATION METRICS
collect_metrics(glmnet_res) %>%
  filter(.metric == "rmse") %>%
  select(penalty, rmse = mean)

# SELECT THE BEST PENALTY
best_penalty <- select_best(glmnet_res, metric = "rmse")

# FIT THE FINAL MODEL WORKFLOW
final_wf <- finalize_workflow(lasso_wf_tune, best_penalty)

final_wf %>% 
  fit(office_train) %>% 
  extract_fit_parsnip() %>% 
  vip::vi(lambda = best_penalty$penalty) %>% 
  mutate(Importance = abs(Importance),
         Variable = fct_reorder(Variable, Importance)) %>% 
  ggplot(aes(x = Importance, y = Variable, fill = Sign)) +
  geom_col() +
  scale_x_continuous(expand = c(0,0)) +
  labs(y = NULL)

# FIT THE FINALISED MODEL TO THE TEST DATA
final_fit <- fit(final_wf, data = office_test)

tidy(final_fit) %>%
  print(n = 50)
