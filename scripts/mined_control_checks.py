"""Require executed tests, not merely a nonzero process exit, for mined controls."""
from pathlib import Path
import xml.etree.ElementTree as ET


def junit_counts(path):
    """Read pytest's JUnit report; missing reports remain unknown."""
    try:
        root = ET.parse(Path(path)).getroot()
    except (OSError, ET.ParseError):
        return None
    suites = [root] if root.tag == 'testsuite' else list(root.findall('testsuite'))
    if not suites:
        return None
    return {key: sum(int(s.get(key, 0)) for s in suites)
            for key in ('tests', 'failures', 'errors', 'skipped')}


def control_verdict(record):
    """Return a conservative acceptance decision and a machine-readable reason."""
    if record.get('error'):
        return {'valid': False, 'reason': 'setup_or_patch_error'}
    for side in ('gold', 'empty'):
        if record.get(side + '_install_rc') not in (None, 0):
            return {'valid': False, 'reason': side + '_install_failed'}
        counts = record.get(side + '_tests')
        if not counts or counts['tests'] <= counts['skipped'] or counts['errors']:
            return {'valid': False, 'reason': side + '_tests_not_executed_cleanly'}
    if record.get('gold_rc') != 0 or record['gold_tests']['failures']:
        return {'valid': False, 'reason': 'gold_not_passing'}
    if record.get('empty_rc') != 1 or record['empty_tests']['failures'] < 1:
        return {'valid': False, 'reason': 'empty_not_a_test_failure'}
    return {'valid': True, 'reason': 'gold_passes_empty_assertions_fail'}
