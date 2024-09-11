import unittest.mock
from Robot_Test_Suite_Runner import RobotTestSuiteRunner
from unittest.mock import patch,mock_open,MagicMock
from openpyxl import Workbook,load_workbook

class TestRobotTestSuiteRunner(unittest.TestCase): 
        
    def testReadingColumn_SameValue(self):                                     #Chechking the ReadingColumn method for return expected data.
        res1 = [8, 'sa', 'Android', 1, '40mhz', 2, 'False', '0.50', '0.80', 4, 2, 'CellMngActDeactWithAttachDetach', 'VWS0220321001005', 'Setup8,DailyDeveloper,CellMngActDeactWithAndroidUEAttachDetach', 'none', '1.0', '0.80', 1, 0]
        res2 = RobotTestSuiteRunner.ReadingColumn(self,"RobotTestSuitesParamsList(deneme).xlsx",2)
        self.assertEqual(res2,res1)

    def testReadingColumn_EmptyData(self):                                     #Chechking the ReadingColumn method for return expected None data.       
        res1 = [None for _ in range(19)]
        res2 = RobotTestSuiteRunner.ReadingColumn(self,"RobotTestSuitesParamsList(deneme).xlsx",4)
        self.assertEqual(res2,res1)
   
    def testReadingColumn_WrongRow(self):                                      #Chechking the ReadingColumn method for return expected Error Handling.
        result1 = RobotTestSuiteRunner.ReadingColumn(self,'RobotTestSuitesParamsList(deneme).xlsx',0)
        result2 = print('Row count must be least 1.') 
        self.assertEqual(result1,result2)

    def testCheckingColumnValue(self):                                         #Chechking the ReadingColumn method for return expected data's index.
        res2 = len(RobotTestSuiteRunner.ReadingColumn(self,"RobotTestSuitesParamsList(deneme).xlsx",2))
        self.assertEqual(res2,19)
        
    def testCheckingEmptyRows(self):                                           #Chechking the EmptyRows method for when send empty datas, return True Bool.  
        list1 = []
        val = RobotTestSuiteRunner.CheckingEmptyRows(self,list1)
        self.assertEqual(val,True)

    def testCheckingRows(self):                                                #Chechking the EmptyRows method for when send not empty datas, return FFalse Bool.   
        list2 = [87,6654,4,'asdaf','dsdsd']
        val = RobotTestSuiteRunner.CheckingEmptyRows(self,list2)
        self.assertFalse(val,False)
        
    @patch('builtins.open', new_callable=mock_open)                            
    def testWriteText(self, mock_open):                                        #Chechking the WriteText method for expected text content .
        Test_TxtFile = "TestFile.txt"
        test_list = ['8', 'sa', 'Android', '1', '40mhz', '2', 'False', '0.50', '0.80', '4', '2', 'CellMngActDeactWithAttachDetach', 'VWS0220321001005', 'Setup8,DailyDeveloper,CellMngActDeactWithAndroidUEAttachDetach', 'none', '1.0', '0.80', '1', '0']
        test_content = (
            '--doc This is a test suite for Setup:8:Mode:sa:UeType:Android:UeNum:1:Freq:40mhz:UeAttachIter:2:Dryrun:False:DLTxCoeff:0.50:DLRxCoeff:0.80:DuTrV:4:CuUpTrV:2:subTestSuiteName:CellMngActDeactWithAttachDetach:androidUeList:VWS0220321001005:TagList:Setup8,DailyDeveloper,CellMngActDeactWithAndroidUEAttachDetach:phyRunArg:none:ULTxCoeff:1.0:ULRxCoeff:0.80:ConfiguredCellNumber:1:ActivatedCellIndexList:0'
            '\n-v Env_Setup_ID:Setup8'
            '\n-v Env_mode:sa'
            '\n-v Env_ue_type:Android'
            '\n-v Env_ue_number:1'
            '\n-v Env_frequency:40mhz'
            '\n-v Env_UeAttachIter:2'
            '\n-v Env_iperf_dl_tx_coeff:0.50'
            '\n-v Env_iperf_dl_rx_coeff:0.80'
            '\n-v Env_du_tr_verbosity:4'
            '\n-v Env_subTestSuiteName:CellMngActDeactWithAttachDetach'
            '\n-v Env_androidUeList:VWS0220321001005'
            '\n-i Setup8'
            '\n-i DailyDeveloper'
            '\n-i CellMngActDeactWithAndroidUEAttachDetach'
            '\n-v Env_pull_request_flag:False'
            '\n-v Env_phyRunArg:none'
            '\n-v Env_iperf_ul_tx_coeff:1.0'
            '\n-v Env_iperf_ul_rx_coeff:0.80'
            '\n-v Env_configured_cell_number:1'
            '\n-v Env_activated_cell_index_list:0'
            '\n--name Setup8saCellMngActDeactWithAttachDetach1AndroidsaUe40mhz'
            '\n-l Setup8saCellMngActDeactWithAttachDetach1AndroidsaUe40mhz_log.html'
            '\n-o Setup8saCellMngActDeactWithAttachDetach1AndroidsaUe40mhz_output.xml'
            '\n-r Setup8saCellMngActDeactWithAttachDetach1AndroidsaUe40mhz_report.html'
            '\n-v Env_reboot:False'
            '\n--exitonfailure'
            '\n-v Env_deploy_directory:/home/ulak/ulak-gnb/fx_2207/bin/nr5g/gnb/DAILY_BUILDS/develop'
            '\n-v Env_cu_run_script:cu_run_robot_develop.sh'
            '\n-v Env_du_run_script:l2_run_robot_develop.sh'
            '\n-v Env_oamc_run_script:oamc_run_robot_develop.sh'
            '\n**********************************************************************************************************************'
        )
        
        RobotTestSuiteRunner.WriteText(self, Test_TxtFile, test_list, 2)       #Send items for test      
        mock_open.assert_called_once_with(Test_TxtFile, 'a')                   #Control that is true text file
        handle = mock_open()
        handle.write.assert_called()        
        called_args = [call[0][0] for call in handle.write.call_args_list]     #Pull the written articles    
        actual_content = ''.join(called_args)                                  
        self.assertEqual(actual_content.strip(), test_content.strip())

    @patch('sys.stdout')
    @patch.object(RobotTestSuiteRunner, 'ReadingColumn')
    @patch.object(RobotTestSuiteRunner, 'CheckingEmptyRows')
    @patch.object(RobotTestSuiteRunner, 'WriteText')
    def testOperateFullData(self, mock_write_text, mock_checking_empty_rows, mock_reading_column, mock_stdout):    #Chechking the Operate method for true process.        
        
        processor = RobotTestSuiteRunner()                                     #Inital values for mock functions
        mock_reading_column.return_value = [1, 2, 3]                           #Sample data
        mock_checking_empty_rows.return_value = True                           #Always not empty row

        processor.Operate('dummy_txt_file.txt', 'dummy_excel_file.xlsx')
        mock_stdout.write.assert_called_with('\n')


      
if __name__ == '__main__':
    unittest.main()

    