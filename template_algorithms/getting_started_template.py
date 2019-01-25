import quantopian.optimize as opt
import quantopian.algorithm as algo
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import factset
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline


def initialize(context):
    # A pipeline that produces alpha factor values which we will use to rebalance our portfolio
    algo.attach_pipeline(make_pipeline(), 'alpha_factors_pipeline')

    # A special pipline provided by Quantopian that helps us avoid key risk factors
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors_pipeline')
  
    # Schedules a function called rebalance to be called once a day, at market open
    algo.schedule_function(
        func=rebalance,
        date_rule=algo.date_rules.every_day(),
        time_rule=algo.time_rules.market_open(),
        half_days=True
    )


def make_pipeline():
    universe = QTradableStocksUS()

    # A pipeline "factor term", which specifies data that we want about each asset in our trading universe 
    net_income = factset.Fundamentals.sales_gr_qf.latest.winsorize(.2, .98).zscore()

    return Pipeline(
        # Pipeline columns usually consists of factor terms
        columns={'net_income': net_income}, 
        # Pipeline screens always consists of filter terms
        screen=universe & net_income.notnull()
    )


def rebalance(context, data):
    # Get the output of our pipelines. 
    pipeline_data = algo.pipeline_output('alpha_factors_pipeline')
    risk_loadings = algo.pipeline_output('risk_factors_pipeline')

    # Convert a column of our dataframe into weight space
    pipeline_column = pipeline_data['net_income']
    weights = pipeline_column/pipeline_column.abs().sum()

    # Our objective is for our portfolio to match the target weights calculated in before_trading_start
    objective = opt.TargetWeights(weights)

    # All of our constraints
    constraints = [
        opt.DollarNeutral(),
        opt.MaxGrossExposure(1),
        opt.PositionConcentration.with_equal_bounds(-.01, .01),
        opt.experimental.RiskModelExposure(
            risk_model_loadings=risk_loadings, 
            version=opt.Newest
        )
    ]

    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )