

# Create your models here.
from django.db import models

class SudokuImage(models.Model):
    image = models.ImageField(upload_to='sudoku_images/')
