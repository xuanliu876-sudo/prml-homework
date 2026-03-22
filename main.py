from regression_utils import load_data
from regression_utils import linear_result, polynomial_result, kernel_result
from regression_utils import plot_linear_fit, plot_polynomial_mse, plot_polynomial_fit
from regression_utils import plot_kernel_bandwidth, plot_kernel_fit
from regression_utils import print_linear_result

def run_task1(X_train, y_train, X_test, y_test):
    result = linear_result(X_train, y_train, X_test, y_test, method="ls")
    print("===== task1 =====")
    print_linear_result(X_train, y_train, X_test, y_test, result)


def run_task2(X_train, y_train, X_test, y_test):
    result = linear_result(X_train, y_train, X_test, y_test, method="gd")
    print("===== task2 =====")
    print_linear_result(X_train, y_train, X_test, y_test, result)


def run_task3(X_train, y_train, X_test, y_test):
    result = linear_result(X_train, y_train, X_test, y_test, method="newton")
    print("===== task3 =====")
    print_linear_result(X_train, y_train, X_test, y_test, result)


def run_task4(X_train, y_train, X_test, y_test):
    result = polynomial_result(X_train, y_train, X_test, y_test)
    print("===== task4 =====")
    for degree, train_mse, test_mse in result["all_results"]:
        print(f"Degree = {degree:2d} | Train MSE = {train_mse:.12f} | Test MSE = {test_mse:.12f}")
    print(f"Best Degree = {result['best_degree']}")
    print(f"Best Test MSE = {result['best_test_mse']:.12f}")
    for i in range(len(result["best_w"])):
        print(f"w{i} = {float(result['best_w'][i, 0]):.12f}")
    print(result["equation"])
    plot_polynomial_mse(result["all_results"])
    plot_polynomial_fit(X_train, y_train, X_test, y_test, result["best_w"], result["best_degree"])


def run_task5(X_train, y_train, X_test, y_test):
    result = kernel_result(X_train, y_train, X_test, y_test)
    print("===== task5 =====")
    print(f"Best bandwidth h = {result['best_h']:.6f}")
    print(f"Train MSE = {result['train_mse']:.6f}")
    print(f"Train RMSE = {result['train_rmse']:.6f}")
    print(f"Train MAE = {result['train_mae']:.6f}")
    print(f"Test MSE = {result['test_mse']:.6f}")
    print(f"Test RMSE = {result['test_rmse']:.6f}")
    print(f"Test MAE = {result['test_mae']:.6f}")
    print(result["result_df"].head(10))
    result["result_df"].to_excel("kernel_regression_test_predictions.xlsx", index=False)
    plot_kernel_bandwidth(result["records"])
    plot_kernel_fit(X_train, y_train, X_test, y_test, result["model"])


def main():
    X_train, y_train, X_test, y_test = load_data("Data4Regression.xlsx")
    choice = input("请输入任务编号(1/2/3/4/5): ").strip()
    if choice == "1":
        run_task1(X_train, y_train, X_test, y_test)
    elif choice == "2":
        run_task2(X_train, y_train, X_test, y_test)
    elif choice == "3":
        run_task3(X_train, y_train, X_test, y_test)
    elif choice == "4":
        run_task4(X_train, y_train, X_test, y_test)
    elif choice == "5":
        run_task5(X_train, y_train, X_test, y_test)
    else:
        print("输入无效")


if __name__ == "__main__":
    main()