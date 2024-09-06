from Robot_Test_Suite_Runner import RobotTestSuiteRunner
import unittest
from unittest.mock import patch
from openpyxl import Workbook,load_workbook

class TestRobotTestSuiteRunner(unittest.TestCase):

    @patch('Robot_Test_Suite_Runner.Operate') 
    def testReadingColumn(self,moc_operate):
        test_dizi = RobotTestSuiteRunner.ReadingColumn(self,2)
        test_mock_dizi = [8, 'sa', 'Android', 1, '40mhz', 2, 'False', '0.50', '0.80', 4, 2, 'CellMngActDeactWithAttachDetach', 'VWS0220321001005', 'Setup8,DailyDeveloper,CellMngActDeactWithAndroidUEAttachDetach', 'none', '1.0', '0.80', 1, 0]
        res1 = self.assertListEqual(test_dizi,test_mock_dizi)
        print(res1)
"""
    def testCheckingEmptyRows(self):
        list1 = []
        val = rob.RobotTestSuiteRunner.CheckingEmptyRows(self,list1)
        res2 = self.assertEqual(val,True)
        print(res2)
    def testWriteText(self):
        dizi = [8, 'sa', 'Android', 1, '40mhz', 2, 'False', '0.50', '0.80', 4, 2, 'CellMngActDeactWithAttachDetach', 'VWS0220321001005', 'Setup8,DailyDeveloper,CellMngActDeactWithAndroidUEAttachDetach', 'none', '1.0', '0.80', 1, 0]
        rob.RobotTestSuiteRunner.WriteText(self,dizi,2)
        test = open("TestArgumentFile.txt","r")
        denek = open("ArgumentFile.txt","r")
        res3 = self.assertEqual(denek,test)
        print(res3)
 """       
if __name__ == '__main__':
    unittest.main()

    