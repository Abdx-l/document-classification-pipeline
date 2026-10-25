from app import route
def test_threshold(tmp_path):
 db=str(tmp_path/"review.db");assert route("x","id",.69,db)["manual_review"];assert not route("x","id",.7,db)["manual_review"]
