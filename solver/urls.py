from django.urls import path
from django.urls import path
from .views import SudokuImageView

urlpatterns = [
    path('solve-sudoku/', SudokuImageView.as_view(), name='solve-sudoku'),
]