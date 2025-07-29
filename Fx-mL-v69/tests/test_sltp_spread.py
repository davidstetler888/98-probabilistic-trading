import importlib
from config import config


def test_sltp_pairs_scale_with_spread(monkeypatch):
    custom_spread = 0.00020
    original_spread = config._config['sl_tp_grid']['spread']
    monkeypatch.setitem(config._config['sl_tp_grid'], 'spread', custom_spread)

    import label
    import sltp
    importlib.reload(label)
    importlib.reload(sltp)

    sl_mults = config.get('sl_tp_grid.sl_multipliers')
    tp_mults = config.get('sl_tp_grid.tp_multipliers')

    expected_grid = [
        (sl * custom_spread * 10000, tp * custom_spread * 10000)
        for sl in sl_mults
        for tp in tp_mults
    ]
    expected_pairs = []
    for sl_mult in sl_mults:
        sl_pips = (custom_spread * sl_mult) / 0.0001
        for tp_mult in tp_mults:
            tp_pips = sl_pips * tp_mult
            if tp_pips >= 2 * sl_pips:
                expected_pairs.append((sl_pips, tp_pips))

    assert sltp.SPREAD == custom_spread
    assert label.create_sltp_grid() == expected_grid
    assert sltp.SL_TP_PAIRS == expected_pairs

    monkeypatch.setitem(config._config['sl_tp_grid'], 'spread', original_spread)
    importlib.reload(label)
    importlib.reload(sltp)
