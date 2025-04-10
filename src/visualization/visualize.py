import matplotlib.pyplot as plt
import seaborn as sns
import logging

def plot_scores(df):
    """
    Plots and saves a scatter plot of GRE vs TOEFL score by Admit_Chance.
    """
    try:
        logging.info("Generating GRE vs TOEFL scatter plot...")
        plt.figure(figsize=(12, 7))
        sns.scatterplot(data=df, x='GRE_Score', y='TOEFL_Score', hue='Admit_Chance', palette='Set1', s=80, alpha=0.8)
        plt.title("GRE vs TOEFL Score by Admission Chance")
        plt.xlabel("GRE Score")
        plt.ylabel("TOEFL Score")
        plt.grid(True)
        plt.legend(title='Admit Chance', loc='best')

        plt.savefig('reports/figures/gre_vs_toefl.png')
        plt.close()
        logging.info("Scatter plot saved to 'reports/figures/gre_vs_toefl.png'.")

    except Exception as e:
        logging.error("Error while plotting GRE vs TOEFL scatter plot", exc_info=True)
        raise

def plot_pairplot(df):
    """
    Creates a pairplot for GRE, TOEFL, CGPA colored by Admit_Chance.
    """
    try:
        logging.info("Generating pairplot for GRE, TOEFL, CGPA...")
        pairplot = sns.pairplot(df, hue="Admit_Chance", vars=["GRE_Score", "TOEFL_Score", "CGPA"], palette="husl")
        pairplot.fig.suptitle("Pairplot of Admission Features", y=1.02)
        pairplot.savefig("reports/figures/pairplot.png")
        plt.close()
        logging.info(f"Pairplot saved to '{"reports/figures/pairplot.png"}'.")

    except Exception as e:
        logging.error("Error while creating pairplot", exc_info=True)
        raise

def plot_cgpa_histogram_by_admission(df):
    """
    Plots a histogram of CGPA, colored by Admit_Chance.
    """
    try:
        logging.info("Generating CGPA histogram by Admit_Chance...")
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='CGPA', hue='Admit_Chance', bins=20, kde=True, palette='Set2', alpha=0.7)
        plt.title("CGPA Distribution by Admission Chance")
        plt.xlabel("CGPA")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("reports/figures/cgpa_hist.png")
        plt.close()
        logging.info(f"CGPA histogram saved to '{"reports/figures/cgpa_hist.png"}'.")

    except Exception as e:
        logging.error("Error while plotting CGPA histogram", exc_info=True)
        raise
