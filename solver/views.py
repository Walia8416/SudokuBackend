from django.shortcuts import render
from rest_framework import generics,status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from .models import SudokuImage
from rest_framework.views import APIView
from .serializers import SudokuImageSerializer
import cv2
import numpy as np
from sudokuDetector import solverNow

class SudokuImageView(APIView):
    queryset = SudokuImage.objects.all()
    serializer_class = SudokuImageSerializer
    parser_classes = (MultiPartParser,)

    def post(self, request, *args, **kwargs):
        serializer = SudokuImageSerializer(data=request.data)
        if serializer.is_valid():
            image_instance = serializer.save()
            image_path = image_instance.image.path

      
            sudoku_matrix = solverNow(image_path)
            image_instance.image.delete()
            return Response({'solved_sudoku': sudoku_matrix}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        
