import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from pandas import DataFrame
from dataclasses import dataclass, field
from typing import Dict, Union, Optional
from collections import defaultdict
import logging

from optimization.BudgetOptimization import (
    SaturationCurve,
    OptimizationConstraint,
    BudgetOptimizer_NLOPT
)

logger = logging.getLogger(__name__)

@dataclass
class PlannerData:
    data: DataFrame = field(default_factory=pd.DataFrame)
    activity_groups: DataFrame = field(default_factory=pd.DataFrame)
    hierarchy: DataFrame = field(default_factory=pd.DataFrame)
    kpis: DataFrame = field(default_factory=pd.DataFrame)
    periods: DataFrame = field(default_factory=pd.DataFrame)
    dataset: DataFrame = field(default_factory=pd.DataFrame)
    constraints_df: DataFrame = field(default_factory=pd.DataFrame)
    df_examples: DataFrame = field(default_factory=pd.DataFrame)
    questionnaire: Dict[str, any] = field(default_factory=dict)
    level_type: Dict[str, list] = field(default_factory=dict)
    driver_mapping: DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def from_local(cls, folder):
        activity_groups = pd.read_csv(folder/'PlannerDataset/activity_groups.csv')
        activity_groups = activity_groups[activity_groups['Base'] != 1].copy()
        hierarchy = pd.read_csv(folder/'PlannerDataset/hierarchy.csv')
        kpis = pd.read_csv(folder/'PlannerDataset/kpis.csv') # merge kpi
        periods = pd.read_csv(folder/"PlannerDataset/periods.csv")
        data = pd.read_csv(folder/'PlannerDataset/data_sum.csv')
        dataset = process_planner_dataset(data, periods, kpis, hierarchy, activity_groups)
        constraints_df = construct_constraint_df(activity_groups, data)
        df_examples = pd.read_csv(folder/'PlannerDataset/planner_in_context_examples.csv')
        questionnaire, level_type = get_planner_questionnaire(folder/'PlannerDataset/')
        driver_mapping = pd.read_csv(folder/'driver_measure_mapping.csv')
        instance = cls(
            data=data,
            activity_groups=activity_groups,
            hierarchy=hierarchy,
            kpis=kpis,
            periods=periods,
            dataset=dataset,
            constraints_df=constraints_df,
            df_examples=df_examples,
            questionnaire=questionnaire,
            level_type=level_type,
            driver_mapping=driver_mapping,
        )
        return instance


@dataclass
class PlannerScenario:
    planner_filter: DataFrame = field(default_factory=pd.DataFrame)
    planner_df: DataFrame = field(default_factory=pd.DataFrame)
    summary_df: DataFrame = field(default_factory=pd.DataFrame)
    response_curve: DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def from_local(cls, clientCode: str, modelgroupId: int):
        planner_filter, planner_df, summary_df, response_curve = get_planner_scenarios(clientCode, modelgroupId)
        instance = cls(
            planner_filter=planner_filter,
            planner_df=planner_df,
            summary_df=summary_df,
            response_curve=response_curve,
        )
        return instance


def get_planner_scenarios(client_code: str, model_group_id: int) -> tuple:
    data_path = f"/datascience6/data/ask-genome-data/{client_code}/{model_group_id}/"
    planner_filter = pd.read_csv(os.path.join(data_path, "planner_filter.csv"))
    planner_df = pd.read_csv(os.path.join(data_path, "planner.csv"))
    summary_df = pd.read_csv(os.path.join(data_path, "planner_scenario_summary.csv"))
    response_curve = pd.read_csv(os.path.join(data_path, "planner_response_curve.csv"))
    planner_filter["Scenario ID"] = planner_filter["Scenario ID"].astype(str)
    planner_df["Scenario ID"] = planner_df["Scenario ID"].astype(str)
    summary_df["Scenario ID"] = summary_df["Scenario ID"].astype(str)
    response_curve["Model ID"] = response_curve["Model ID"].astype(str)

    percent_columns = [col for col in planner_df.columns if '%' in col]
    planner_df[percent_columns] = planner_df[percent_columns].replace('#DIV/0!', np.inf)
    return planner_filter, planner_df, summary_df, response_curve


def get_planner_questionnaire(folder) -> tuple[dict, dict]:
    """
    get planner questionnaire config
    """
    questionnaire_df = pd.read_csv(folder/'planner_questionnaire.csv', low_memory=False)
    questionnaire = questionnaire_df.fillna('').set_index('dimension')[['field', 'context', 'prompt']].T.to_dict()
    for _, v in questionnaire.items():
        v['field'] = [x.strip() for x in v['field'].split(",")]

    if 'level' not in questionnaire_df.columns:
        questionnaire_df['level'] = ''
    else:
        questionnaire_df['level'].fillna('', inplace=True)
    level_type_df = questionnaire_df[questionnaire_df['level'] != ''][['field', 'level']]
    # level_type = level_type_df.groupby('level_type').agg(list)['field'].to_dict()
    level_type = defaultdict(list)
    for i, row in level_type_df.iterrows():
        level = [x.strip() for x in row['level'].split(',')]
        field = [x.strip() for x in row['field'].split(',')]
        for lvl, f in zip(level, field):
            level_type[lvl].append(f)
    return questionnaire, level_type


def process_planner_dataset(data, periods, kpis, hierarchy, activity_groups):
    data['kpiValue'] = data['kpiValue'].fillna(0)
    dataset = pd.merge(data, periods, left_on='PeriodId', right_on='Id', how='left').merge(hierarchy, left_on='HierarchyId', right_on='Id', how='left').merge(activity_groups, left_on='ActivityGroupId', right_on='Id', how='left').merge(kpis, left_on='KpiId', right_on='ID', how='left')
    dataset['curveCoefC'] = dataset['curveCoefC'].fillna(dataset['proxyCurveCoefC'])
    dataset['curveCoefD'] = dataset['curveCoefD'].fillna(dataset['proxyCurveCoefD'])
    dataset['AvgWeeklySupportAMP'] = dataset['proxyAvgWeeklySupportAMP']
    dataset['WeeksOnAir'] = dataset['proxyWeeksOnAir']
    dataset['RetentionRate'] = dataset['proxyRetentionRate']
    columns = ['Activity Group', 'Product','AggregateName','spend', 'KpiName', 'kpiValue','activity','WeeksOnAir','RetentionRate','curveCoefC', 'curveCoefD', 'AvgWeeklySupportAMP']
    dataset = dataset[columns].copy().query("kpiValue!=0")
    dataset['enable_curve'] = True
    dataset.loc[dataset['curveCoefC'].isnull(),'enable_curve'] = False
    return dataset


def construct_constraint_df(activity_groups: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    valid_driver_ids = data['ActivityGroupId'].unique()
    constraint_cols = ['Id', 'Base', 'Activity Group', 'LowerBound', 'UpperBound']
    constraint_df = activity_groups[constraint_cols].copy()
    constraint_df = constraint_df[(constraint_df['Base']!=1) & (constraint_df['Id'].isin(valid_driver_ids))]
    constraint_df = constraint_df.rename(columns={'Id': 'ID', 'Activity Group': 'Driver',
                                                  'LowerBound': 'Minimum Spend',
                                                  'UpperBound': 'Maximum Spend'})
    constraint_df.drop_duplicates(inplace=True)
    constraint_df.set_index('Driver', inplace=True)
    constraint_df['Minimum Spend'] *= 100
    constraint_df['Maximum Spend'] *= 100
    constraint_df['Constraint Type (Percentage/Absolute)'] = 'Percentage'

    return constraint_df


def create_channels(dataset, period, product, kpi):
    model_dataset = dataset.query(f"AggregateName=='{period}'").query(f"Product=='{product}'").query(f"KpiName=='{kpi}'").copy()
    channels = {}
    for _, row in model_dataset.iterrows():
        if isinstance(row['Activity Group'], str):
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
        else:
            pass
    return channels


def get_llm_answers(folder, query):
    llm_answers = pd.read_csv(folder/'llm_answers.csv')
    llm_answers['query'] = llm_answers['query'].str.lower()
    answer_dict = llm_answers.query("query==@query").to_dict('records')
    return answer_dict[0]


def process_llm_answer(ans_dict, planner_data, field_types):
    res_dict = {}
    for key, val in ans_dict.items():
        if key == 'budget_change':
            val = convert_to_float_or_keep(val)
            res_dict[key] = val
        elif key in planner_data.columns:
            # find key df and found correct value
            field_type = field_types.get(key, 'not available')
            if hasattr(planner_data, field_type):
                field_df = getattr(planner_data, field_type)
            if val == 'irrelevant':
                # select the one with min levelId TODO: need to check with processed data
                # TODO: rename columns when load data
                val = field_df.loc[field_df['LevelId'].idxmin()][key]
                res_dict[key] = val
            elif val == 'all':
                pass
            elif val in field_df[key].values:
                res_dict[key] = val
            else:
                pass
        else:
            res_dict[key] = val


def single_channel_simulation(curve, spend: float, label: str) -> dict:
    # response = curve.calc_response(spend)
    roi = curve.calculate_roi(spend)
    margin_roi = curve.calculate_marginal_roi(spend)
    contribution = curve.calculate_contribution(spend)
    res = {f'{label} spend': spend,
           # f'{label} response': response,
           f'{label} roi': roi,
           f'{label} margin roi': margin_roi,
           f'{label} contribution': contribution}
    return res


def convert_to_float_or_keep(s):
    s = s.strip().replace(',', '')
    if s.endswith('%'):
        try:
            return float(s[:-1]) / 100
        except ValueError:
            return s
    try:
        return float(s)
    except ValueError:
        return s


def calculate_target_spend_by_curve(curve, budget_change: Union[str, float], target_channels: list) -> float:
    """
    calcuate target spend given budget change condition and current spend
    """
    spend = curve.spend
    # budget_change = convert_to_float_or_keep(budget_change)
    # if len(target_channels) and isinstance(budget_change, float):
    #     budget_change /= len(target_channels) # evenly change target spend by channel
    budget_change_percent = 1
    target_spend = spend

    if budget_change == 'fixed':
        target_spend = curve.calculate_optimal_spend()
    elif isinstance(budget_change, str):
        if budget_change == 'increase':
            budget_change_percent = 1.2
        elif budget_change == 'decrease':
            budget_change_percent = 0.8
        target_spend = spend * budget_change_percent
    elif isinstance(budget_change, float): # decrease by 20% -0.2 decrease to 20% -0.8
        if abs(budget_change) < 1:
            budget_change_percent = budget_change
        elif abs(budget_change) > 1:
            budget_change_percent = budget_change / spend
        target_spend = spend * (1 + budget_change_percent)
    return target_spend


def simulation(channels, budget_change: Union[str, float], target_channels: list) -> pd.DataFrame:
    if not target_channels:
        target_channels = list(channels.keys())
    res = []
    for channel, curve in channels.items():
        curve = channels[channel]
        current_spend = curve.spend
        channel_res = {'channel': channel}
        current_res = single_channel_simulation(curve, current_spend, 'current')

        if channel in target_channels:
            target_spend = calculate_target_spend_by_curve(curve, budget_change, target_channels)
            target_res = single_channel_simulation(curve, target_spend, 'target')
        else:
            # target_spend = calculate_target_spend_by_curve(curve, budget_change, target_channels)
            target_res = single_channel_simulation(curve, current_spend, 'target')

        optimal_spend = calculate_target_spend_by_curve(curve, 'fixed', target_channels)
        optimal_res = single_channel_simulation(curve, optimal_spend, 'optimal')
        channel_res.update(current_res)
        channel_res.update(target_res)
        channel_res.update(optimal_res)
        res.append(channel_res)
    return pd.DataFrame(res)


def modify_constraints_df(constraints_df: pd.DataFrame, channels, budget_change: Union[str, float], target_channels: list) -> pd.DataFrame:
    if not target_channels:
        target_channels = list(channels.keys())
    spend = sum([val.spend for key, val in channels.items() if key in target_channels])
    target_constraints_df = constraints_df[constraints_df.index.isin(target_channels)].copy()
    non_target_constrains_df = constraints_df[~constraints_df.index.isin(target_channels)].copy()

    if budget_change == 'fixed':
        pass
    elif budget_change == 'increase':
        target_constraints_df['Minimum Spend'] = 100
        target_constraints_df['Maximum Spend'] = 120
    elif budget_change == 'decrease':
        target_constraints_df['Minimum Spend'] = 80
        target_constraints_df['Maximum Spend'] = 100
    elif isinstance(budget_change, float): # decrease by 20% -0.2 decrease to 20% -0.8
        if abs(budget_change) < 1:
            budget_change_percent = budget_change
        elif abs(budget_change) > 1:
            budget_change_percent = budget_change / spend
        min_spend, max_spend = min(1+budget_change_percent, 1), max(1+budget_change_percent, 1)
        target_constraints_df['Minimum Spend'] = min_spend * 100
        target_constraints_df['Maximum Spend'] = max_spend * 100
    modified_constraints_df = pd.concat([target_constraints_df, non_target_constrains_df])
    modified_constraints_df.drop_duplicates(inplace=True)
    return modified_constraints_df


def create_bundle_constraints(channels, budget_change: Union[str, float], target_channels: list):
    bundle_constraints = []

    if not target_channels:
        return None

    bundle_spend = sum([curve.spend for channel, curve in channels.items() if channel in target_channels])

    if budget_change == 'fixed':
        bundle_constraints.append(OptimizationConstraint(target_channels, bundle_spend, bundle_spend))
    elif budget_change == 'increase':
        bundle_constraints.append(OptimizationConstraint(target_channels, bundle_spend, bundle_spend * 1.2))
    elif budget_change == 'decrease':
        bundle_constraints.append(OptimizationConstraint(target_channels, bundle_spend * 0.8, bundle_spend))
    elif isinstance(budget_change, float):
        if abs(budget_change) < 1:
            budget_change_percent = budget_change
        elif abs(budget_change) > 1:
            budget_change_percent = budget_change / bundle_spend
        else:
            budget_change_percent = 1
        min_spend, max_spend = min(1+budget_change_percent, 1), max(1+budget_change_percent, 1)
        bundle_constraints.append(OptimizationConstraint(target_channels, bundle_spend * min_spend, bundle_spend * max_spend))

    return bundle_constraints


def calculate_total_budget(channels, budget_change, constraints_df, target_channels) -> float:
    if target_channels:
        pass
    spend = sum([curve.spend for channel, curve in channels.items()])
    ratio = 1
    if budget_change == 'fixed':
        ratio = 1
    elif budget_change == 'increase':
        ratio = 1.2
    elif budget_change == 'decrease':
        ratio = 0.8
    elif isinstance(budget_change, float):
        if abs(budget_change) < 1:
            ratio = 1 + budget_change
        elif abs(budget_change) > 1:
            ratio = 1 + budget_change / spend
    return ratio * spend


def optimization(channels, constraints_df, bundle_constraints=None, total_budget=None) -> pd.DataFrame:
    # current (simulation)
    sim_df = simulation(channels, 'fixed', list(channels.keys()))
    # optimization
    Optimizer = BudgetOptimizer_NLOPT(channels)
    opt_df = Optimizer.optimize(scenario='OptimizeToSpend',
                                constraints_df=constraints_df,
                                bundle_constraints=bundle_constraints,
                                total_budget=total_budget)
    opt_df = opt_df.reset_index(drop=False).rename(columns={'index': 'channel',
                                                            'spend': 'optimized spend',
                                                            'contribution': 'optimized contribution',
                                                            'roi': 'optimized roi'})
    # combine
    current_cols = ['channel', 'current spend', 'current roi', 'current contribution']
    combined_df = sim_df[current_cols].merge(opt_df, on='channel')
    return combined_df


def run_planner(clientCode: str, modelgroupId: int, conditions: dict, planner_data: PlannerData=None) -> tuple:
    """
    run planner optimization given mapped conditions
    """
    # find exist scenario
    plannerScenario = PlannerScenario.from_local(clientCode, modelgroupId)
    matched_scenario = find_exist_scenario(planner_conditions=conditions, plannerScenario=plannerScenario)
    # target channels
    target_channels = conditions.get("target_channels", [])

    if matched_scenario:
        planner_result = plannerScenario.planner_df.query(f"`Scenario ID` == '{matched_scenario}'")
        summary_df = plannerScenario.summary_df.query(f"`Scenario ID` == '{matched_scenario}'")
        response_curve = plannerScenario.response_curve.query(f"`Model ID` == '{matched_scenario}'")
        conditions["target_kpi"] = plannerScenario.planner_filter.query(f"`Scenario ID` == '{matched_scenario}'")["target_kpi"].values[0]
    else:
        # load planner data
        if planner_data is None:
            folder = Path(f"/datascience6/data/ask-genome-data/{clientCode}/{modelgroupId}/")
            planner_data = PlannerData.from_local(folder)
        res_df = pd.DataFrame()

        # filter dataset and create channels
        period = conditions.get("time")
        product = conditions.get("product")[0]
        kpi = conditions.get("kpi")

        channels = create_channels(dataset=planner_data.dataset, period=period, product=product, kpi=kpi)

        if not channels:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), conditions

        target_channels = [x for x in target_channels if x in channels.keys()]

        # budget change
        budget_change = conditions["budget_change"]
        budget_change_converted = convert_to_float_or_keep(s=budget_change)

        if conditions['operation'] == "simulation":
            res_df = simulation(channels, budget_change_converted, target_channels)
        elif conditions['operation'] == "optimization":
            # calculate total budget
            # modify constraints df
            # create bundle constraint
            constraints_df = modify_constraints_df(planner_data.constraints_df, channels, budget_change_converted, target_channels)
            bundle_constraints = create_bundle_constraints(channels, budget_change_converted, target_channels)
            total_budget = calculate_total_budget(channels, budget_change_converted, constraints_df, target_channels)
            res_df = optimization(channels, constraints_df, bundle_constraints, total_budget)


        planner_result = format_planner_result(planner_result=res_df, conditions=conditions)
        summary_df = summarize_planner_result(planner_result=planner_result, conditions=conditions)
        response_curve = calculate_response_curve(channels=channels, conditions=conditions)

        if target_channels:
            response_curve = response_curve[response_curve['Driver'].isin(target_channels)].reset_index(drop=True).copy()
        else:
            response_curve = response_curve[response_curve['Driver'].isin(["Total"])].reset_index(drop=True).copy()

    if target_channels:
        planner_result = planner_result[planner_result['Driver'].isin(target_channels)].reset_index(drop=True).copy()
    return planner_result, summary_df, response_curve, conditions


def get_time_answer_dict(answer_dict: dict, data: pd.DataFrame) -> dict:
    """
    select most recent time based on period type
    """
    period = answer_dict.get("period", "irrelevant")
    time = answer_dict.get("time", "irrelevant")
    res_dict = {}

    if period in ["year", "fiscal year"]:
        period = "Year"
    elif period in ["quarter", "fiscal quarter"]:
        period = "Quarter"
    else:
        period = data.sort_values(by=["AggregateSortOrder"], ascending=True)["LevelName"].values[0]

    time = data[data["LevelName"]==period].sort_values(by=["AggregateSortOrder"], ascending=False)["AggregateName"].values[0]

    res_dict["period_type"] = period
    res_dict["time"] = time
    return res_dict


def get_hierarchy_answer_dict(answer_dict: dict, data: pd.DataFrame) -> dict:
    """
    select hierarchy as list
    """
    product = answer_dict.get("product", "irrelevant")
    res_dict = {}
    available_hierarchy = data[data["Enabled"]==1]["Product"].unique()

    if isinstance(product, str):
        product = product.split(",")

    if any(x for x in product if x in available_hierarchy):
        product = [x for x in product if x in available_hierarchy]
    else:
        product = [data.sort_values(by=["LevelId", "SortOrder"], ascending=True)["Product"].values[0]]

    res_dict["product"] = product
    return res_dict


def get_kpi_answer_dict(answer_dict: dict, data: pd.DataFrame) -> dict:
    kpi = answer_dict.get("kpi", "irrelevant")
    res_dict = {}
    available_kpi = data["KpiName"].unique()

    if kpi in available_kpi:
        pass
    else:
        kpi = data.sort_values(by=["SortOrder"], ascending=True)["KpiName"].values[0]

    res_dict["kpi"] = kpi
    res_dict["target_kpi"] = kpi + " ROI"
    return res_dict

def generate_planner_response(conditions, planner_df, summary_df, response_curve) -> str:
    operation = conditions["operation"]
    target_channels = conditions.get("core_dimension", {}).keys()
    kpi = conditions.get("kpi")
    scenario_id = planner_df["Scenario ID"].unique()[0]
    scenario_description = planner_df["Scenario Description"].unique()[0]
    response = ""
    if target_channels:
        response += f"For {', '.join(target_channels)} drivers:  \n"
    response += "Spend - Historical: " + str(
        summary_df[summary_df["KPI"] == 'Total Spend']["Historical"].values[0]) + "  \n"
    if operation == "simulation":
        response += "Spend - Forecast: " + str(
            summary_df[summary_df["KPI"] == 'Total Spend']["Forecast"].values[0]) + "  \n"
        response += f"{kpi} - Forecast: " + str(
            summary_df[summary_df["KPI"] == kpi]["Forecast"].values[0]) + "  \n"
    elif operation == "optimization":
        response += "Spend - Optimized: " + str(
            summary_df[summary_df["KPI"] == 'Total Spend']["Forecast"].values[0]) + "  \n"
        response += f"{kpi} - Optimized: " + str(
            summary_df[summary_df["KPI"] == kpi]["Forecast"].values[0]) + "  \n"
    if scenario_id != "na":
        response += f"Results are from existing scenario {scenario_id}, {scenario_description}"
    return response


def format_planner_result(planner_result: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    res_df = planner_result.copy()
    # rename columns
    target_kpi = conditions.get("kpi", "Kpi")
    renamings = {"channel": "Driver",
                 "current spend": "Spend - Historical", "target spend": "Spend - Forecast",
                 "current contribution": f"{target_kpi} - Historical", "target contribution": f"{target_kpi} - Forecast",
                 "current roi": f"{target_kpi} ROI - Historical", "target roi": f"{target_kpi} ROI - Forecast",
                 "optimized spend": "Spend - Optimized", "optimized contribution": f"{target_kpi} - Optimized",
                 "optimal spend": "Spend - Optimal", "optimal contribution": f"{target_kpi} - Optimal",
                 "optimized roi": f"{target_kpi} ROI - Optimized", "optimal roi": f"{target_kpi} ROI - Optimal",
                 }
    res_df = res_df.rename(columns=renamings)

    # add condition columns
    res_df["Scenario ID"] = conditions.get("scenario_id", "0")
    res_df["Scenario Description"] = generate_scenario_description(conditions)
    res_df["Time Period"] = conditions.get("time", "irrelevant")

    # select columns
    if conditions.get("operation", "simulation") == "simulation":
        res_df["Spend Incremental"] = res_df["Spend - Forecast"] - res_df["Spend - Historical"]
        res_df[f"{target_kpi} Incremental"] = res_df[f"{target_kpi} - Forecast"] - res_df[f"{target_kpi} - Historical"]

        res_df[f"{target_kpi} ROI Incremental"] = res_df[f"{target_kpi} ROI - Forecast"] - res_df[
            f"{target_kpi} ROI - Historical"]

        condition_columns = ["Spend - Historical", "Spend - Forecast", "Spend Incremental", "Spend % Change",
                             f"{target_kpi} - Historical", f"{target_kpi} - Forecast",
                             f"{target_kpi} Incremental", f"{target_kpi} % Change",
                             f"{target_kpi} ROI - Historical", f"{target_kpi} ROI - Forecast",
                             f"{target_kpi} ROI Incremental", f"{target_kpi} ROI % Change",
                             "Spend - Optimal", f"{target_kpi} - Optimal", f"{target_kpi} ROI - Optimal"]
    else:
        res_df["Spend Incremental"] = res_df["Spend - Optimized"] - res_df["Spend - Historical"]
        res_df[f"{target_kpi} Incremental"] = res_df[f"{target_kpi} - Optimized"] - res_df[f"{target_kpi} - Historical"]
        res_df[f"{target_kpi} ROI Incremental"] = res_df[f"{target_kpi} ROI - Optimized"] - res_df[
            f"{target_kpi} ROI - Historical"]

        condition_columns = ["Spend - Historical", "Spend - Optimized", "Spend Incremental", "Spend % Change",
                             f"{target_kpi} - Historical", f"{target_kpi} - Optimized",
                             f"{target_kpi} Incremental", f"{target_kpi} % Change",
                             f"{target_kpi} ROI - Historical", f"{target_kpi} ROI - Optimized",
                             f"{target_kpi} ROI Incremental", f"{target_kpi} ROI % Change"]

    res_df[f"Spend % Change"] = np.where(
        res_df[f"Spend - Historical"] == 0,
        np.inf,
        (res_df[f"Spend Incremental"] / res_df[f"Spend - Historical"]).round(2) * 100
    )

    res_df[f"{target_kpi} ROI % Change"] = np.where(
        res_df[f"{target_kpi} ROI - Historical"] == 0,
        np.inf,
        (res_df[f"{target_kpi} ROI Incremental"] / res_df[f"{target_kpi} ROI - Historical"]).round(2) * 100
    )
    res_df[f"{target_kpi} % Change"] = np.where(
        res_df[f"{target_kpi} - Historical"] == 0,
        np.inf,
        (res_df[f"{target_kpi} Incremental"] / res_df[f"{target_kpi} - Historical"]).round(2) * 100
    )

    display_columns = ["Scenario ID", "Scenario Description", "Time Period",
                       "Driver"] + condition_columns
    res_df = res_df[display_columns]
    return res_df


def generate_scenario_description(conditions: dict) -> str:
    base_description = conditions.get("description", "")
    if base_description:
        return base_description
    else:
        level_condition, target_channel_condition, budget_condition = "", "", ""
        target_channel_condition = "for " + ", ".join(conditions.get("channel", [])) + " drivers" if conditions.get("channel", []) else ""
        budget_change = conditions.get("budget_change")
        budget_change = convert_to_float_or_keep(s=budget_change)
        if budget_change == "fixed":
            budget_condition = "remained the same"
        elif budget_change == 'increase':
            budget_condition = "increased by 20%"
        elif budget_change == 'decrease':
            budget_condition = "decreased by 20%"
        elif isinstance(budget_change, float):
            if budget_change < 0:
                budget_condition = f"decreased by "
            else:
                budget_condition = f"increased by "
            if abs(budget_change) < 1:
                budget_change = str(np.round(budget_change * 100)) + "%"
            else:
                budget_change = "$" + str(budget_change)
            budget_condition += budget_change
        if conditions.get("product", ""):
            level_condition = "for product " + ', '.join(conditions.get("product", []))
        base_description = "the spend budget"
        if target_channel_condition:
            base_description += " " + target_channel_condition
        if budget_condition:
            base_description += " " + budget_condition
        if level_condition:
            base_description += " " + level_condition
    return base_description


def summarize_planner_result(planner_result: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    """
    calculate planner result summary
    """
    target_kpi = conditions.get("kpi", "Kpi")
    operation = conditions.get("operation", "")

    # columns
    spend_columns = [x for x in planner_result.columns if 'Spend - ' in x]
    kpi_columns = [x for x in planner_result.columns if target_kpi + ' - ' in x and 'ROI' not in x]
    roi_columns = [x for x in planner_result.columns if target_kpi in x and 'ROI - ' in x]
    rename_historical = {x: "Historical" for x in spend_columns + kpi_columns + roi_columns if "Historical" in x}
    rename_historical.update({x: "Forecast" for x in spend_columns + kpi_columns + roi_columns if "Forecast" in x})
    rename_historical.update({x: "Optimal" for x in spend_columns + kpi_columns + roi_columns if "Optimal" in x})
    rename_historical.update({x: "Forecast" for x in spend_columns + kpi_columns + roi_columns if "Optimized" in x})

    spend_info = planner_result.groupby(["Scenario ID", "Scenario Description", "Time Period"], as_index=False)[
        spend_columns].sum().rename(columns=rename_historical)
    kpi_info = planner_result.groupby(["Scenario ID", "Scenario Description", "Time Period"], as_index=False)[
        kpi_columns].sum().rename(columns=rename_historical)
    roi_info = planner_result.groupby(["Scenario ID", "Scenario Description", "Time Period"], as_index=False)[
        roi_columns].sum().rename(columns=rename_historical)

    roi_info["Historical"] = kpi_info["Historical"] / spend_info["Historical"]
    forecast_column = "Forecast"

    roi_info[forecast_column] = kpi_info[forecast_column] / spend_info[forecast_column]
    # roi_info["Optimal"] = kpi_info["Optimal"] / spend_info["Optimal"]

    summary_info = pd.concat([spend_info, kpi_info, roi_info]).reset_index(drop=True)
    summary_info["KPI"] = ["Total Spend", target_kpi, f"{target_kpi} ROI"]
    summary_info["Growth %"] = ((summary_info[forecast_column] - summary_info["Historical"]) * 100 / summary_info[
        "Historical"]).round(2).astype(str) + "%"
    summary_info["Historical Period"] = summary_info[f"{forecast_column} Period"] = summary_info["Time Period"]
    summary_info["Historical"] = "$" + summary_info["Historical"].astype(str)
    summary_info[forecast_column] = "$" + summary_info[forecast_column].astype(str)

    summary_info = summary_info[["Scenario ID", "Scenario Description", "KPI",
                                 "Historical", forecast_column, "Growth %",
                                 "Historical Period", f"{forecast_column} Period"]]
    return summary_info


def find_exist_scenario(planner_conditions: dict, plannerScenario: PlannerScenario):
    operation = planner_conditions.get("operation", "")
    budget_change = planner_conditions.get("budget_change", "")
    product = planner_conditions.get("product", [])
    time = planner_conditions.get("time", "")
    driver = planner_conditions.get("driver", "")
    kpi = planner_conditions.get("kpi", "")

    df = plannerScenario.planner_filter.copy()
    filtered_df = pd.DataFrame()
    if all(x for x in ["operation", "budget_change", "product", "time", "kpi"] if x in df.columns):
        filtered_df = df[
            (df['operation'] == operation) &
            (df['budget_change'] == budget_change) &
            (df['product'].isin(product)) &
            (df['time'] == time) &
            (df['kpi'] == kpi)
            ]
    if not filtered_df.empty:
        scenario_id = filtered_df.iloc[0]['Scenario ID']
        logger.info(f"Matching scenario {scenario_id} found with conditions {planner_conditions}.")
        return scenario_id
    else:
        logger.info(f"No matching scenario found with conditions {planner_conditions}.")
        return None


def calculate_response_curve(channels, conditions: dict, n: int=200) -> pd.DataFrame:
    scenario_id = conditions.get("scenario_id", "0")
    target_kpi = conditions.get("kpi", "Margin")
    cumulative_spend, cumulative_contribution = np.zeros(n), np.zeros(n)
    delta_spends, delta_contributions = np.zeros(n), np.zeros(n)
    single_channel_response_curves = []
    for channel_name, channel in channels.items():
        # calculate spend
        channel_spend, channel_contribution, channel_delta_spend, channel_delta_contribution = channel.calculate_response_curve(points=n)
        cumulative_spend += channel_spend
        cumulative_contribution += channel_contribution
        delta_spends += channel_delta_spend
        delta_contributions += channel_delta_contribution

        channel_roi_curve = channel_contribution / channel_spend
        channel_marginal_roi_curve = channel_delta_contribution / channel_delta_spend

        single_channel_response_curves.append(pd.DataFrame({"Model ID": scenario_id,
                                               "Driver": channel_name,
                                               "Spend": channel_spend,
                                               f"Average Margin ROI - {target_kpi}": channel_roi_curve,
                                               f"Marginal Margin ROI - {target_kpi}": channel_marginal_roi_curve,
                                               f"Cumulative KPI - {target_kpi}": channel_contribution,
                                               }))

    roi_curve = cumulative_contribution / cumulative_spend
    marginal_roi_curve = delta_contributions / delta_spends

    total_df = pd.DataFrame({"Model ID": scenario_id,
                           "Driver": "Total",
                           "Spend": cumulative_spend,
                           f"Average Margin ROI - {target_kpi}": roi_curve,
                           f"Marginal Margin ROI - {target_kpi}": marginal_roi_curve,
                           f"Cumulative KPI - {target_kpi}": cumulative_contribution,
                           })
    single_channel_response_curves_df = pd.concat(single_channel_response_curves)
    res_df = pd.concat([single_channel_response_curves_df, total_df]).reset_index(drop=True)

    return res_df