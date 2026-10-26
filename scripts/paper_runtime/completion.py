"""Stop eligibility uses only the delivered hypothesis and learned pose estimate."""
def shared_stop_eligible(snapshot):
    d=snapshot['decision']
    return (d['state'] in ('execute','verify') and snapshot['goal'] is not None
            and snapshot['goal_symbol'] is not None and d['color']==snapshot['goal'])
