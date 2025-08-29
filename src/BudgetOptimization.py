import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from scipy.optimize import differential_evolution, LinearConstraint, Bounds, minimize
from dataclasses import dataclass
import nlopt
import pandas as pd
from pathlib import Path
# curve parameters
# spend, activity, retention_rate, weeksonair, AMP, c, d.

class SaturationCurve:
    def __init__(self, spend:float, activity:float, contribution:float, retention_rate:float, weeksonair:int, AMPWeeklySupport:float, c:float, d:float, enable_curve=True):
        self.spend = spend
        self.activity = activity
        self.contribution = contribution
        self.ROI = self.contribution / self.spend
        self.unitcost = self.spend / self.activity
        self.enable_curve = enable_curve
        if enable_curve:
            self.retention_rate = retention_rate
            self.weeksonair = weeksonair
            self.AMPWeeklySupport = AMPWeeklySupport
            self.weekly_contribution = self.contribution / self.weeksonair
            # curve parameters not can not be changed after initialization
            # c ranges from -0.0009 to -0.0001
            # d ranges from 1.1 to 1.9
            if not (-0.0009 <= c <= -0.0001):
                raise ValueError("Parameter c must be between -0.0009 and -0.0001")
            if not (1.0 <= d <= 1.9):
                raise ValueError("Parameter d must be between 1.1 and 1.9")
            self.c = c
            self.d = d
            # calculation
            # can not be changed after initialization
            self.scaled_factor = self.activity / self.weeksonair / self.AMPWeeklySupport
            self.current_scaled_support = self.AMPWeeklySupport/(1-self.retention_rate)
            self.current_saturation_level = self.saturation_level(self.current_scaled_support)
            self.current_response = self.calc_response(self.current_scaled_support)
            self.response_index = 100
            #add features to account for channels without curves


    def saturation_level(self, scaled_support):
        return 1 - np.exp(self.c * scaled_support ** self.d)

    def calc_response(self, scaled_support):
        return self.saturation_level(scaled_support) / scaled_support


    def calc_scaled_support(self, spend, weeksonair, unitcost, retention_rate):
        impressions = spend / unitcost
        # effective impressions
        effective_impressions = impressions / (1-retention_rate)
        # weekly effective impressions
        weekly_effective_impressions = effective_impressions / weeksonair
        # Calculate the saturation effect
        scaled_support = weekly_effective_impressions / self.scaled_factor
        return scaled_support

    # Core function
    def calculate_contribution(self, spend, weeksonair=None, unitcost=None, retention_rate=None, response_index=None):
        if self.enable_curve:
            if weeksonair is None:
                weeksonair = self.weeksonair
            if unitcost is None:
                unitcost = self.unitcost
            if retention_rate is None:
                retention_rate = self.retention_rate
            if response_index is None:
                response_index = self.response_index

            media_support = spend / unitcost
            scaled_support = self.calc_scaled_support(spend, weeksonair, unitcost, retention_rate)
            bm_scaled_support = self.calc_scaled_support(self.spend, self.weeksonair, unitcost, retention_rate)
            bm_response = self.calc_response(bm_scaled_support)
            # calculate weekly contribution first then adjust to whole year contribution
            # response = self.calc_response(scaled_support)
            if unitcost != self.unitcost:
                response_adj = self.current_response / bm_response
            else:
                response_adj = 1
            contribution = self.saturation_level(scaled_support) / self.current_saturation_level * self.weekly_contribution * weeksonair * response_adj * response_index / self.response_index
        else:
            contribution = spend * self.ROI
        return contribution


    def calculate_roi(self, spend, weeksonair = None, unitcost = None, retention_rate=None, response_index=None):
        if spend <= 0:
            return 0.0  # Return 0 ROI for zero or negative spend
        contribution = self.calculate_contribution(spend, weeksonair, unitcost, retention_rate, response_index)
        return contribution / spend
    
    def calculate_optimal_spend(self, weeksonair = None, unitcost = None, retention_rate=None):
        """Find the spend that maximizes ROI using grid search followed by local optimization"""
        # First do a grid search to find approximate maximum
        min_spend = max(self.spend * 0.1, 1.0)  # Ensure minimum spend is at least 1
        max_spend = self.spend * 2
        grid_points = 1000

        while True:
            end_roi = self.calculate_roi(max_spend)
            end_marginal_roi = self.calculate_marginal_roi(max_spend)
            if end_roi > end_marginal_roi:
                break
            else:
                min_spend, max_spend = max_spend, max_spend * 2
        
        spends = np.linspace(min_spend, max_spend, grid_points)
        rois = [self.calculate_roi(spend, weeksonair, unitcost, retention_rate) for spend in spends]
        best_grid_spend = spends[np.argmax(rois)]
        
        # Use minimize starting from the best grid point for fine-tuning
        def objective(spend):
            if spend[0] <= 0:
                return float('inf')  # Penalize non-positive spend
            return -self.calculate_roi(spend[0], weeksonair, unitcost, retention_rate)
        
        result = minimize(objective, [best_grid_spend], bounds=[(1.0, None)], method='L-BFGS-B')
        return result.x[0]

    def calculate_marginal_roi(self, spend, weeksonair = None, unitcost = None, retention_rate = None):
        """Calculate marginal ROI as the derivative of ROI with respect to spend"""
        if self.enable_curve:
            delta = 0.0001
            contribution_plus = self.calculate_contribution(spend + delta)
            contribution_minus = self.calculate_contribution(spend - delta)
            marginal_roi = (contribution_plus - contribution_minus) / (2 * delta)
        else:
            marginal_roi = 0
        return marginal_roi

    def plot_roi_curve(self, min_spend=None, max_spend=None, points=100):
        """
        Plot the ROI curve and marginal ROI curve over a range of spend values.
        
        Args:
            min_spend (float): Minimum spend to plot (default: 20% of current spend)
            max_spend (float): Maximum spend to plot (default: 300% of current spend)
            points (int): Number of points to plot
        """
        if min_spend is None:
            min_spend = max(self.spend * 0.2, 1.0)
        if max_spend is None:
            max_spend = self.spend * 10
        
        # Ensure min_spend is positive
        min_spend = max(min_spend, 1.0)
        
        spends = np.linspace(min_spend, max_spend, points)
        rois = [self.calculate_roi(spend) for spend in spends]
        marginal_rois = [self.calculate_marginal_roi(spend) for spend in spends]
        
        plt.figure(figsize=(12, 6))
        
        # Plot both curves on same axis
        plt.plot(spends, rois, 'b-', label='ROI Curve')
        plt.plot(spends, marginal_rois, 'g-', label=f'Marginal ROI')
        
        # Plot current spend point
        current_roi = self.calculate_roi(self.spend)
        plt.plot(self.spend, current_roi, 'bo', label=f'Current Spend: ${self.spend/1000000:.2f}MM, current ROI: {current_roi:.2f}')
        
        # Plot optimal spend point
        optimal_spend = self.calculate_optimal_spend()
        optimal_roi = self.calculate_roi(optimal_spend)
        plt.plot(optimal_spend, optimal_roi, 'ro', label=f'Optimal Spend: ${optimal_spend/1000000:.2f}MM, Optimal ROI: {optimal_roi:.2f}')
        
        # Add zero line
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        
        plt.title('ROI and Marginal ROI vs Spend')
        plt.xlabel('Spend ($)')
        plt.ylabel('ROI / Marginal ROI')
        plt.grid(True)
        plt.legend(loc='upper right')
        
        return 0


    def calculate_response_curve(self, min_spend=None, max_spend=None, points=100):
        """
        Calculate the ROI curve and marginal ROI curve over a range of spend values.

        Args:
            min_spend (float): Minimum spend to plot (default: 10% of current spend)
            max_spend (float): Maximum spend to plot (default: 200% of current spend)
            points (int): Number of points to plot
        """
        if min_spend is None:
            min_spend = max(self.spend * 0.1, 1.0)
        if max_spend is None:
            max_spend = self.spend * 2

        # Ensure min_spend is positive
        min_spend = max(min_spend, 1.0)

        spends = np.linspace(min_spend, max_spend, points)
        contributions = np.array([self.calculate_contribution(spend) for spend in spends])
        rois = np.array([self.calculate_roi(spend) for spend in spends])
        marginal_rois = np.array([self.calculate_marginal_roi(spend) for spend in spends])

        # delta spend
        delta = 1e-5
        delta_spend = np.array([2 * delta * spend for spend in spends])
        delta_contributions = np.array([self.calculate_contribution(spend * (1+delta)) - self.calculate_contribution(spend * (1-delta)) for spend in spends])

        return spends, contributions, delta_spend, delta_contributions


@dataclass
class OptimizationConstraint:
    channels: List[str]
    min_budget: float
    max_budget: float



class BudgetOptimizer:
    def __init__(self, channels: Dict):
        """Initialize BudgetOptimizer with channel curves.

        Args:
            channels (Dict): Dictionary mapping channel names to their SaturationCurve objects
        """
        self.channels = channels
        self.channel_list = list(channels.keys())
        self.total_budget = sum(channel.spend for channel in channels.values())
        self._validate_channels()

    def _validate_channels(self) -> None:
        """Validate channel data for optimization."""
        if not self.channels:
            raise ValueError("No channels provided for optimization")

        for name, curve in self.channels.items():
            if not hasattr(curve, 'calculate_contribution'):
                raise ValueError(f"Channel {name} missing required method: calculate_contribution")
            if not hasattr(curve, 'spend'):
                raise ValueError(f"Channel {name} missing required attribute: spend")

    def _create_bounds(self, min_percent:float=0.8, max_percent:float=1.2) -> Bounds:
        """Create bounds for optimization.

        Args:
            min_percent (float): Minimum percentage of current spend (default: 0.8)
            max_percent (float): Maximum percentage of current spend (default: 1.2)

        Returns:
            Bounds: Scipy optimization bounds
        """
        #
        lb = [channel.spend * min_percent for channel in self.channels.values()]
        ub = [channel.spend * max_percent for channel in self.channels.values()]
        return Bounds(lb, ub)

    def _create_budget_constraint(self, constraints: List[OptimizationConstraint]) -> List[LinearConstraint]:
        """Create budget constraints for optimization.

        Args:
            constraints (List[OptimizationConstraint]): List of budget constraints

        Returns:
            List[LinearConstraint]: List of Scipy linear constraints
        """
        linear_constraints = []

        for constraint in constraints:
            array = np.zeros(len(self.channels))
            positions = [i for i, channel in enumerate(self.channel_list)
                         if channel in constraint.channels]
            array[positions] = 1
            linear_constraints.append(
                LinearConstraint(array.reshape(1, -1),
                                 constraint.min_budget,
                                 constraint.max_budget)
            )

        return linear_constraints

    def _objective_function(self, x: np.ndarray) -> float:
        """Optimization objective function maximizing overall ROI.

        Args:
            x (np.ndarray): Array of spend values

        Returns:
            float: Negative ROI (for minimization)
        """
        contributions = [curve.calculate_contribution(x[i])
                         for i, curve in enumerate(self.channels.values())]
        return -sum(contributions) / sum(x)  # Negative for minimization

    def optimize(self,
                 constraints: Optional[List[OptimizationConstraint]] = None,
                 min_percent: float = 0.8,
                 max_percent: float = 1.2,
                 maxiter: int = 5,
                 tol: float = 0.01) -> Tuple[Dict[str, float], float]:
        """Optimize budget allocation across channels.

        Args:
            constraints (List[OptimizationConstraint], optional): Budget constraints
            min_percent (float): Minimum percentage of current spend
            max_percent (float): Maximum percentage of current spend
            maxiter (int): Maximum iterations for optimization
            tol (float): Tolerance for optimization convergence

        Returns:
            Tuple[Dict[str, float], float]: Optimized allocations and overall ROI
        """
        bounds = self._create_bounds(min_percent, max_percent)

        if constraints is None:
            # Default constraint: maintain total budget
            constraints = [OptimizationConstraint(
                channels=self.channel_list,
                min_budget=self.total_budget,
                max_budget=self.total_budget
            )]

        linear_constraints = self._create_budget_constraint(constraints)

        result = differential_evolution(
            self._objective_function,
            bounds=bounds,
            constraints=linear_constraints[0] if len(linear_constraints) == 1 else linear_constraints,
            maxiter=maxiter,
            disp=False,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            tol=tol
        )

        # result = shgo(
        #     self._objective_function,
        #     bounds=bounds,
        #     constraints=linear_constraints[0] if len(linear_constraints) == 1 else linear_constraints,
        #     iters=maxiter,
        # )

        # if not result.success:
        #     raise RuntimeError(f"Optimization failed: {result.message}")

        # Calculate final results
        optimized_spends = dict(zip(self.channel_list, result.x))
        contributions = [curve.calculate_contribution(spend)
                         for curve, spend in zip(self.channels.values(), result.x)]
        overall_roi = sum(contributions) / sum(result.x)

        return optimized_spends, overall_roi

    def get_channel_metrics(self, allocation: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for given budget allocation.

        Args:
            allocation (Dict[str, float]): Channel spend allocation

        Returns:
            Dict[str, Dict[str, float]]: Channel-wise metrics
        """
        metrics = {}
        for channel, spend in allocation.items():
            curve = self.channels[channel]
            contribution = curve.calculate_contribution(spend)
            metrics[channel] = {
                'spend': spend,
                'contribution': contribution,
                'roi': contribution / spend if spend > 0 else 0
            }
        return metrics



class BudgetOptimizer_NLOPT:
    """
    Comprehensive marketing channel optimization class.
    """

    def __init__(self, channels: Dict):
        """
        Initialize the optimizer with dataset and constraints.

        Args:
            optimization_dataset (pd.DataFrame): Dataset with channel performance metrics
            constraints_path (str): Path to constraints CSV file
        """
        # Initialize channels
        self.channels = channels

        # Calculate total budget
        self.total_budget = sum(curve.spend for curve in self.channels.values())
        # initialize channel bounds
        self.constraints_df = None
        # Initial spend guess
        self.initial_spend = [curve.spend for curve in self.channels.values()]

    def _calculate_channel_bounds(self, constraints_df) -> Tuple[List[float], List[float]]:
        """
        Calculate lower and upper bounds for each channel based on constraints.

        Returns:
            Tuple of lower and upper bounds lists
        """
        lower_bounds, upper_bounds = [], []
        for channel, curve in self.channels.items():
            constraint_type = constraints_df.loc[channel, 'Constraint Type (Percentage/Absolute)']
            min_value = constraints_df.loc[channel, 'Minimum Spend']
            max_value = constraints_df.loc[channel, 'Maximum Spend']

            if constraint_type == 'Percentage':
                min_spend = curve.spend * min_value / 100
                max_spend = curve.spend * max_value / 100
            else:
                min_spend = min_value
                max_spend = max_value

            lower_bounds.append(min_spend)
            upper_bounds.append(max_spend)
        return lower_bounds, upper_bounds

    def _objective_function(self, x: np.ndarray, grad: np.ndarray) -> float:
        """
        Objective function to maximize total contribution.

        Args:
            x (np.ndarray): Channel spends
            grad (np.ndarray): Gradient array

        Returns:
            float: Total normalized contribution
        """
        if grad.size > 0:
            grad[:] = [
                (list(self.channels.values())[i].calculate_marginal_roi(x[i]) * x.sum() -
                 list(self.channels.values())[i].calculate_contribution(x[i])) / x.sum() ** 2
                for i in range(len(self.channels))
            ]

        return sum([
            list(self.channels.values())[i].calculate_contribution(x[i])
            for i in range(len(self.channels))
        ]) / x.sum()

    def _objective_function_kpi_target(self, x:np.ndarray, grad: np.ndarray) -> float:
        if grad.size > 0:
            grad[:] = [
                (list(self.channels.values())[i].calculate_marginal_roi(x[i]) * x.sum() -
                 list(self.channels.values())[i].calculate_contribution(x[i])) / x.sum() ** 2
                for i in range(len(self.channels))
            ]
        return x.sum()

    def _budget_constraint(self, x, grad:np.ndarray, total_budget, isupper=True):
        if grad.size > 0:
            grad[:] = np.ones(x.shape)
        if isupper:
            return x.sum() - total_budget
        else:
            return total_budget - x.sum()

    def _contribution_constraint(self, x: np.ndarray, grad: np.ndarray, target_kpi: float, isupper=False) -> float:
        """
        Constraint for minimum contribution.

        Args:
            x (np.ndarray): Channel spends
            grad (np.ndarray): Gradient array
            target_kpi (float): Target Key Performance Indicator

        Returns:
            float: Contribution difference from target
        """
        if grad.size > 0:
            grad[:] = [
                list(self.channels.values())[i].calculate_marginal_roi(x[i])
                for i in range(len(self.channels))
            ]
        if isupper:
            return sum([
                list(self.channels.values())[i].calculate_contribution(x[i])
                for i in range(len(self.channels))
            ]) - target_kpi
        else:

            return -sum([
                list(self.channels.values())[i].calculate_contribution(x[i])
                for i in range(len(self.channels))
            ]) + target_kpi

    # vector constraints
    def _bundle_constraints(self, results: np.ndarray, x: np.ndarray, grad: np.ndarray,
                            constraint_list: List[OptimizationConstraint], is_upper: bool = True) -> None:
        """
        Apply bundle constraints to optimization.

        Args:
            results (np.ndarray): Constraint results array
            x (np.ndarray): Channel spends
            grad (np.ndarray): Gradient array
            constraint_list (List[OptimizationConstraint]): List of bundle constraints
            is_upper (bool): Flag to determine upper or lower constraint
        """
        channel_names = list(self.channels.keys())

        for n, constraints in enumerate(constraint_list):
            # Create an array of zeros with 1s for relevant channels
            array = np.zeros(len(self.channels))
            positions = [channel_names.index(channel) for channel in constraints.channels]
            array[positions] = 1

            # Calculate constraint value
            bundle_spend = (x * array).sum()
            results[n] = (bundle_spend - constraints.max_budget) if is_upper else (
                        -bundle_spend + constraints.max_budget)
            # Set gradient if needed
            if grad.size > 0:
                grad[n, :] = array[:]

    def optimize(self, scenario: str = 'OptimizeToSpend', target_kpi: Optional[float] = None,
                 total_budget: float = None,
                 bundle_constraints: Optional[List[OptimizationConstraint]] = None, constraints_df: pd.DataFrame = None,
                 lower_bounds_default=0.8, upper_bounds_default=1.2, tol=1e-10):
        """
        Perform marketing channel optimization.

        Args:
            scenario (str): Optimization scenario type: OptimizeToSpend, OptimizeToKPITarget
            target_kpi (float, optional): Target KPI for specific scenarios
            bundle_constraints (List[OptimizationConstraint], optional): Bundle constraints

        Returns:
            pd.DataFrame: Optimization results
        """
        # Initialize optimizer LN_COBYA
        alg = nlopt.LN_COBYLA
        # alg = nlopt.GN_ISRES
        opt = nlopt.opt(alg, len(self.channels))
        opt.set_max_objective(self._objective_function)
        # if scenario =='OptimizeToSpend':
        #     opt.set_max_objective(self._objective_function)
        # else:
        #     opt.set_min_objective(self._objective_function_kpi_target)

        opt.set_xtol_abs(np.ones(len(self.channels)) * tol)
        opt.set_ftol_rel(tol)
        opt.set_maxeval(10000)
        # Prepare bounds
        if constraints_df is not None:
            lower_bounds, upper_bounds = self._calculate_channel_bounds(constraints_df)
        else:
            lower_bounds = [curve.spend * lower_bounds_default for _, curve in self.channels.items()]
            upper_bounds = [curve.spend * upper_bounds_default for _, curve in self.channels.items()]

        # Set bounds
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        # total_budget_constraint
        # Initialize total budget
        if total_budget is None:
            total_budget = self.total_budget

        # Add constraints based on scenario
        if scenario == 'OptimizeToSpend':
            opt.add_inequality_constraint(lambda x, grad: self._budget_constraint(x, grad, total_budget, True), tol)
            opt.add_inequality_constraint(lambda x, grad: self._budget_constraint(x, grad, total_budget, False), tol)

        elif scenario == 'OptimizeToKPITarget' and target_kpi is not None:
            # opt.add_inequality_constraint(
            #     lambda k, grad: self._contribution_constraint(k, grad, target_kpi, True),
            #     tol
            # )
            opt.add_inequality_constraint(
                lambda k, grad: self._contribution_constraint(k, grad, target_kpi, False),
                tol
            )
        # Add bundle constraints if available.
        if bundle_constraints is not None:
            opt.add_inequality_mconstraint(
                lambda results, x, grad: self._bundle_constraints(results, x, grad, bundle_constraints, True),
                np.ones(len(bundle_constraints)) * tol
            )
            opt.add_inequality_mconstraint(
                lambda results, x, grad: self._bundle_constraints(results, x, grad, bundle_constraints, False),
                np.ones(len(bundle_constraints)) * tol
            )

        # Optimize
        initial_guess = lower_bounds
        results = opt.optimize(initial_guess)

        # Parse and return results
        return self._parse_results(results)

    def _parse_results(self, spends: np.ndarray) -> pd.DataFrame:
        """
        Parse optimization results into a DataFrame.

        Args:
            spends (np.ndarray): Optimized channel spends

        Returns:
            pd.DataFrame: Detailed optimization metrics
        """
        metrics = {}
        for i, (channel, curve) in enumerate(self.channels.items()):
            spend = spends[i]
            contribution = curve.calculate_contribution(spend)
            metrics[channel] = {
                'spend': spend,
                'contribution': contribution,
                'roi': contribution / spend if spend > 0 else 0
            }

        return pd.DataFrame(metrics).T


def process_planner_dataset(folder):
    activity_groups = pd.read_csv(folder/'activity_groups.csv')
    activity_groups = activity_groups[activity_groups['Base']!=1].copy()
    hierarchy = pd.read_csv(folder/'hierarchy.csv')
    kpis = pd.read_csv(folder/'kpis.csv')
    periods = pd.read_csv(folder/"periods.csv")
    data = pd.read_csv(folder/'data.csv')
    dataset = pd.merge(data, periods, left_on='PeriodId', right_on='Id', how='left').merge(hierarchy, left_on='HierarchyId', right_on='Id', how='left').merge(activity_groups, left_on='ActivityGroupId', right_on='Id', how='left')
    columns = ['Activity Group', 'Product','AggregateName','spend','kpiValue','activity','WeeksOnAir','RetentionRate','curveCoefC', 'curveCoefD', 'AvgWeeklySupportAMP']
    dataset = dataset[columns].copy().query("kpiValue!=0")
    dataset['enable_curve'] = True
    dataset.loc[dataset['curveCoefC'].isnull(),'enable_curve'] = False
    return dataset


def create_channels(dataset, period, product):
    period = 'FY24Q1'
    product = '.COM*'
    model_dataset = dataset.query(f"AggregateName=='{period}'").query(f"Product=='{product}'").copy()
    channels = {}
    for _, row in model_dataset.iterrows():
        channel = SaturationCurve(
            spend = row['spend'],
            activity = row['activity'],
            contribution = row['kpiValue'],
            retention_rate = row['RetentionRate'],
            weeksonair=row['WeeksOnAir'],
            AMPWeeklySupport=row['AvgWeeklySupportAMP'],
            c = row['curveCoefC'],
            d = row['curveCoefD'],
            enable_curve=row['enable_curve']
        )
        channels[row['Activity Group']] = channel
    return channels



def main():
    DATAPATH = Path.cwd() / "PlannerDataset"
    dataset = process_planner_dataset(folder=DATAPATH)
    period = 'FY24Q1'
    product = '.COM*'
    channels = create_channels(dataset=dataset, period=period, product=product)
    Optimizer = BudgetOptimizer_NLOPT(channels)
    constraints_df = pd.read_csv(DATAPATH / 'constrain template.csv', index_col='Driver')
    constraints_df['Minimum Spend'] = 80
    constraints_df['Maximum Spend'] = 100
    results = Optimizer.optimize(scenario='OptimizeToSpend', constraints_df=constraints_df)
    print(results)

if __name__ == '__main__':
    main()
