import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from visual_question_answering_claim_audit.audit import audit, has_evidence, phrase_matches
from visual_question_answering_claim_audit.phrases import VQA, NOVQA, GENERIC_NAMES

POSITIVE = [
'visual question answering', 'answers questions about images', 'answer questions about images',
'answers questions about an image', 'answer questions about an image', 'visual question answerer',
'visual question answering system', 'visual question answering feature', 'visual question answering support',
'visual question answering capability', 'supports visual question answering', 'provides visual question answering',
'performs visual question answering', 'answers image questions', 'answer image questions', 'image question answering',
'answers questions on images', 'answer questions on images', 'question answering on images', 'question answering for images',
'answers visual questions', 'answer visual questions', 'vqa system', 'vqa feature', 'vqa support', 'vqa capability',
'supports vqa', 'provides vqa', 'performs vqa', 'image vqa', 'visual qa', 'answers questions about pictures',
'answer questions about pictures', 'answers questions about photos', 'answer questions about photos',
'picture question answering', 'photo question answering', 'answers picture questions', 'answers photo questions',
'question answering over images', 'answers questions from images', 'answer questions from images',
'image-based question answering', 'question answering with images', 'answers image-based questions', 'vqa model',
'visual question answering model', 'answers a question about an image', 'answer a question about an image',
'question answering on an image']
NEGATIVE = [
'does not answer questions about images', 'do not answer questions about images', "doesn't answer questions about images",
"don't answer questions about images", 'no visual question answering', 'without visual question answering',
'never answers questions about images', 'does not answer image questions', 'no visual question answerer',
'without a visual question answerer', 'visual question answering not supported', 'visual question answering unsupported',
'no visual question answering support', 'does not answer visual questions', 'never answers image questions',
'lacks visual question answering', 'missing visual question answering', 'visual question answering disabled',
'no vqa', 'without vqa', 'does not provide vqa', 'no question answering on images', 'vqa not supported',
'does not answer questions about pictures']
NAMES = ['answer_question', 'answer_visual_question', 'visual_question_answering', 'vqa_answer',
'ask_question_about_image', 'answer_image_question', 'answer_about_image', 'get_answer', 'get_visual_answer',
'predict_answer', 'answer_question_for_image', 'question_answering_on_image', 'vqa', 'answer_vqa',
'generate_answer', 'generate_visual_answer', 'answer_image', 'ask_image_question', 'query_image', 'answer_query',
'visual_answer', 'image_question_answer', 'answer_with_image', 'respond_to_image_question']


class AuditTests(unittest.TestCase):
    def fixture(self, files):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for name, content in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding='utf-8')
        return root

    def cli(self, root, *args):
        return subprocess.run([sys.executable, '-m', 'visual_question_answering_claim_audit', 'audit', str(root), *args], capture_output=True, text=True)

    def test_phrase_oracle(self):
        self.assertEqual(len(POSITIVE), 50)
        self.assertEqual(len(NEGATIVE), 24)
        self.assertEqual(VQA, POSITIVE)
        self.assertEqual(NOVQA, NEGATIVE)

    def test_generic_names_oracle(self):
        self.assertEqual(len(NAMES), 24)
        self.assertEqual(GENERIC_NAMES, NAMES)

    def test_pipeline_routes(self):
        for source in [
            'import transformers\ntransformers.pipeline("visual-question-answering")',
            'import transformers as tf\ntf.pipeline("visual-question-answering")',
            'from transformers import pipeline\npipeline("visual-question-answering")',
            'from transformers import pipeline as make\nmake(task="visual-question-answering")',
            'import transformers\ntransformers.pipeline(task="visual-question-answering")',
        ]:
            with self.subTest(source=source):
                self.assertTrue(has_evidence(source))

    def test_pair_routes(self):
        for model, call in [('BlipForQuestionAnswering', 'm.generate()'), ('ViltForQuestionAnswering', 'm()'), ('LxmertForQuestionAnswering', 'm()')]:
            for constructor in ['M()', 'M.from_pretrained("id")']:
                with self.subTest(model=model, constructor=constructor):
                    self.assertTrue(has_evidence('from transformers import ' + model + ' as M\nm = ' + constructor + '\n' + call))
            self.assertTrue(has_evidence('import transformers as t\nm = t.' + model + '()\nm.generate()'))
            self.assertTrue(has_evidence('import transformers as t\nt.' + model + '().generate()'))

    def test_generic_routes(self):
        for name in NAMES:
            for source in ['x.' + name + '()', 'from external import ' + name + '\n' + name + '()', 'from external import ' + name + ' as run\nrun()']:
                with self.subTest(source=source):
                    self.assertTrue(has_evidence(source))

    def test_rejections(self):
        sources = [
            'from transformers import pipeline, BlipForQuestionAnswering',
            'from external import answer_question', 'answer_question', 'answer_question()',
            '"visual question answering"\n# x.answer_question()\n"""x.vqa()"""',
            'import pytesseract\npytesseract.image_to_string(x)\npytesseract.image_to_data(x)',
            'import easyocr\nr=easyocr.Reader([])\nr.readtext(x)',
            'import paddleocr\npaddleocr.PaddleOCR().ocr(x)',
            'from transformers import BlipForConditionalGeneration\nm=BlipForConditionalGeneration()\nm.generate()',
            'x.caption_image()', 'classifier=ImageNet()\nclassifier(x)', 'clip.encode_image(x)',
            'detector.detect(x)', 'segmenter.segment(x)', 'diffusion(prompt)',
            'from transformers import AutoModelForQuestionAnswering\nm=AutoModelForQuestionAnswering()\nm(text)',
            'import subprocess\nsubprocess.run(["vqa", "image"])',
            'm.generate()',
            'import transformers\ntask="visual-question-answering"\ntransformers.pipeline(task)',
        ]
        for task in ['image-to-text', 'image-classification', 'question-answering', 'text-generation', 'object-detection', 'image-segmentation']:
            sources.append('from transformers import pipeline\npipeline(' + repr(task) + ')')
        for model in ['BlipForQuestionAnswering', 'ViltForQuestionAnswering', 'LxmertForQuestionAnswering']:
            sources.append('from transformers import ' + model + '\nm=' + model + '()')
            sources.append('from transformers import ' + model + '\nm=' + model + '()\nother.generate()')
        sources.append('from transformers import BlipForQuestionAnswering\nm=BlipForQuestionAnswering()\nm()')
        for source in sources:
            with self.subTest(source=source):
                self.assertFalse(has_evidence(source))

    def test_shadowing(self):
        for source in [
            'from external import answer_question\ndef answer_question(): pass\nanswer_question()',
            'def answer_question(): pass\nx.answer_question()',
            'from external import answer_question as aq\ndef aq(): pass\naq()',
            'import transformers as tf\nclass tf: pass\ntf.pipeline("visual-question-answering")',
            'from transformers import pipeline\ndef pipeline(x): pass\npipeline("visual-question-answering")',
        ]:
            self.assertFalse(has_evidence(source))
        for model in ['BlipForQuestionAnswering', 'ViltForQuestionAnswering', 'LxmertForQuestionAnswering']:
            self.assertFalse(has_evidence('from transformers import ' + model + '\nclass ' + model + ': pass\nm=' + model + '()\nm.generate()\nm()'))

    def test_split_modules(self):
        root = self.fixture({'README.md': POSITIVE[0], 'a.py': 'from transformers import BlipForQuestionAnswering\nm=BlipForQuestionAnswering()', 'b.py': 'm.generate()'})
        self.assertEqual(len(audit(root)[0]), 1)

    def test_span_removal(self):
        self.assertEqual(audit(self.fixture({'README.md': NEGATIVE[0]}))[0], [])
        for phrase in NEGATIVE:
            self.assertEqual(audit(self.fixture({'README.md': phrase}))[0], [])

    def test_longest_first(self):
        self.assertEqual(phrase_matches('visual question answering system', VQA)[0], ['visual question answering system'])

    def test_fences_and_inline(self):
        self.assertEqual(audit(self.fixture({'README.md': '```\nvisual question answering\n```'}))[0], [])
        self.assertEqual(len(audit(self.fixture({'README.md': '`visual question answering`'}))[0]), 1)

    def test_exact_matching(self):
        self.assertEqual(len(phrase_matches('VISUAL QUESTION ANSWERING visual question answering', VQA)[0]), 2)
        self.assertEqual(phrase_matches('visual-question-answering', VQA)[0], [])

    def test_good_pass(self):
        result = self.cli(self.fixture({'README.md': POSITIVE[0], 'app.py': 'x.answer_question()'}))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'PASS')

    def test_bad_high_json(self):
        result = self.cli(self.fixture({'README.md': POSITIVE[0]}), '--json')
        self.assertEqual(result.returncode, 2)
        finding, = json.loads(result.stdout)
        self.assertEqual(finding['rule'], 'documented-visual-question-answering-claim-false')
        self.assertEqual(finding['verdict'], 'HIGH')
        self.assertTrue(finding['message'])

    def test_test_only_medium(self):
        root = self.fixture({'tests/fixture.py': repr(POSITIVE[0])})
        result = self.cli(root, '--json')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)[0]['verdict'], 'MEDIUM')
        self.assertEqual(self.cli(root, '--fail-on', 'medium').returncode, 2)

    def test_denial_high(self):
        result = self.cli(self.fixture({'docs/usage.md': NEGATIVE[0], 'app.py': 'x.answer_question()'}), '--json')
        self.assertEqual(result.returncode, 2)
        finding, = json.loads(result.stdout)
        self.assertEqual(finding['rule'], 'documented-no-visual-question-answering-claim-false')
        self.assertEqual(finding['verdict'], 'HIGH')

    def test_no_claim(self):
        self.assertEqual(self.cli(self.fixture({'README.md': 'Hello.'})).stdout.strip(), 'PASS')

    def test_version(self):
        result = subprocess.run([sys.executable, '-m', 'visual_question_answering_claim_audit', '--version'], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), '0.1.0')

    def test_ignore_and_verbose(self):
        root = self.fixture({'docs/a.md': POSITIVE[0], 'extra/b.py': 'x.answer_question()'})
        self.assertEqual(self.cli(root, '--ignore', 'extra/').returncode, 2)
        result = self.cli(root, '--ignore', 'docs/', '--ignore', 'extra/', '--verbose')
        self.assertEqual(result.returncode, 0)
        self.assertIn('Scan scope:', result.stderr)

    def test_usage_and_io_errors(self):
        root = self.fixture({'bad.py': 'def broken('})
        self.assertEqual(self.cli(root).returncode, 1)
        self.assertEqual(self.cli(root / 'missing').returncode, 1)
        self.assertEqual(self.cli(root, '--fail-on', 'low').returncode, 1)

    def test_exclusions(self):
        root = self.fixture({'README.md': POSITIVE[0], '.git/a.py': 'x.answer_question()', 'custom/pyvenv.cfg': '', 'custom/a.py': 'x.answer_question()'})
        self.assertEqual(len(audit(root)[0]), 1)

    def test_readme_self_audit(self):
        readme = Path(__file__).resolve().parents[1] / 'README.md'
        self.assertTrue(readme.is_file())
        self.assertEqual(audit(self.fixture({'README.md': readme.read_text(encoding='utf-8')}))[0], [])


if __name__ == '__main__':
    unittest.main()
