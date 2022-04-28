Requirements have packages that are not required for this code. It's generated using an environment used to develop other ML projects and might have extra packages.

The codes available in the src folder were run on a system with 32GB ram and 6GB of cuda available GPU and might require some minor code changes to run in a cloud environment like Google collab (File paths etc)
decrease in batch size for neural nets or the min_df param for TF-IDF vectorizer if RAM available is low.

The output generated from the Logistic regression model is saved as 'simple_logistic_regression_test_output_predictions.csv' under the data folder.

The saved models available might not load correctly and generate outputs as they were stopped mid training due to resource limitations. The 'generate_output_final_test.py' file is a template that's work in progress
that we want to use to generate outputs from our saved trained models that's mentioned in the future work.

The data preprocessing code requires packages like nltk and nlpaug which should be installed using the requirements file available.

The presentation and report of the project is stored under the doc folder.