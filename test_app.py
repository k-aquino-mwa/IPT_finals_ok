import unittest
from app import app, db
from models import Document

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app = app.test_client()
        with app.app_context():
            db.create_all()
            doc = Document(text="Sample document for testing")
            db.session.add(doc)
            db.session.commit()
            self.sample_doc_id = doc.id

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def test_list_documents_page(self):
        response = self.app.get('/documents')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Documents', response.data)

    def test_create_document_page(self):
        response = self.app.get('/documents/new')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Create Document', response.data)

    def test_edit_document_page(self):
        response = self.app.get(f'/documents/{self.sample_doc_id}/edit')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Edit Document', response.data)
        self.assertIn(b'Sample document for testing', response.data)

if __name__ == '__main__':
    unittest.main()
