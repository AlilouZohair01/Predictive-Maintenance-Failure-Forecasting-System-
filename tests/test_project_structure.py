"""Checks for notebook relocation; no training or optional ML imports required."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / 'notebooks').glob('*.ipynb'))


class ProjectStructureTests(unittest.TestCase):
    def test_notebook_sources_compile_and_use_current_paths(self):
        self.assertEqual(len(NOTEBOOKS), 6)
        obsolete = ('dataset/archive/CMaps', 'processed data/', 'saved_models/',
                    "Path('outputs/", 'nasa_rul_lstm_v1.keras')
        for path in NOTEBOOKS:
            nb = json.loads(path.read_text(encoding='utf-8'))
            for number, cell in enumerate(nb['cells'], 1):
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source'])
                with self.subTest(notebook=path.name, cell=number):
                    compile(source, f'{path.name}:cell{number}', 'exec')
                    for old in obsolete:
                        self.assertNotIn(old, source)

    def test_project_paths_from_both_working_directories(self):
        for path in NOTEBOOKS:
            nb = json.loads(path.read_text(encoding='utf-8'))
            cell = next(c for c in nb['cells'] if c.get('id') == 'project-paths')
            source = ''.join(cell['source'])
            source += '\nassert PROJECT_ROOT == Path(' + repr(str(ROOT)) + ')\n'
            for cwd in [ROOT, ROOT / 'notebooks']:
                with self.subTest(notebook=path.name, cwd=str(cwd)):
                    result = subprocess.run([sys.executable, '-c', source], cwd=cwd,
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_current_artifacts_and_reference_files_exist(self):
        required = [
            'data/processed/lstm_fd001/X_train.npy',
            'data/processed/lstm_fd001/X_val.npy',
            'data/processed/ml_baselines/df_clean_FD001.csv',
            'models/lstm_fd001_v2/model.keras',
            'models/lstm_fd001_v2/best_model.keras',
            'models/lstm_fd001_v2/scaler.pkl',
            'models/cox_fd001/cox_bundle.pkl',
            'reports/figures/lstm_fd001_v2/shap_engine55_heatmap.png',
            'reports/figures/survival_fd001/conditional_survival.png',
            'reports/tables/survival_fd001/failure_probabilities.csv',
            'references/cmapss_readme.txt',
        ]
        for subset in range(1, 5):
            required.extend(f'data/raw/cmapss/{kind}_FD00{subset}.txt'
                            for kind in ['train', 'test', 'RUL'])
        for relative_path in required:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).is_file())


if __name__ == '__main__':
    unittest.main()
