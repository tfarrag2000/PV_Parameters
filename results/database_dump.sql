-- MySQL dump 10.13  Distrib 8.0.35, for Win64 (x86_64)
--
-- Host: localhost    Database: pv_forecasting
-- ------------------------------------------------------
-- Server version	8.0.35

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `experiment_results`
--

DROP TABLE IF EXISTS `experiment_results`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `experiment_results` (
  `id` int NOT NULL AUTO_INCREMENT,
  `experiment_ID` varchar(255) NOT NULL,
  `test_RMSE` float NOT NULL,
  `test_MAPE` float NOT NULL,
  `median_ABE` float DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `experiment_ID` (`experiment_ID`),
  CONSTRAINT `experiment_results_ibfk_1` FOREIGN KEY (`experiment_ID`) REFERENCES `experiments` (`experiment_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `experiment_results`
--

LOCK TABLES `experiment_results` WRITE;
/*!40000 ALTER TABLE `experiment_results` DISABLE KEYS */;
/*!40000 ALTER TABLE `experiment_results` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `experiments`
--

DROP TABLE IF EXISTS `experiments`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `experiments` (
  `id` int NOT NULL AUTO_INCREMENT,
  `experiment_ID` varchar(255) NOT NULL,
  `n_epochs` int NOT NULL,
  `n_batch` int NOT NULL,
  `outputIndex` int NOT NULL,
  `ANN_arch` varchar(255) NOT NULL,
  `Dropout` float NOT NULL,
  `earlystop` tinyint(1) NOT NULL,
  `RMSE` float NOT NULL,
  `MAPE` float NOT NULL,
  `min_train_loss` float NOT NULL,
  `min_val_loss` float NOT NULL,
  `Model_summary` text,
  `comment` text,
  `optimizer` varchar(255) NOT NULL,
  `ActivationFunctions` varchar(255) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `experiment_ID` (`experiment_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=63 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `experiments`
--

LOCK TABLES `experiments` WRITE;
/*!40000 ALTER TABLE `experiments` DISABLE KEYS */;
INSERT INTO `experiments` VALUES (1,'20231030124839',88,256,-1,'[32, 64, 256, 256, 5, 5]',0,1,1.42397,9.88274,0.003976,0.004557,'None','test','Adam','tanh'),(2,'20231030125110',500,256,-1,'[32, 64, 256, 256, 5, 5]',0,1,1.30306,6.4273,0.002619,0.003243,'None','test','Adam','tanh'),(3,'20231030125509',317,128,-1,'[256, 8, 8, 8, 256, 5]',0,1,1.41355,7.16207,0.003256,0.003821,'None','model_Major3: output:-1 generation:0 index:0 count:0','adam','tanh'),(4,'20231030125718',445,256,-1,'[256, 256, 512, 256, 8, 5]',0,1,1.20437,6.17285,0.002284,0.002799,'None','model_Major3: output:-1 generation:0 index:1 count:0','adam','tanh'),(5,'20231030130037',572,32,-1,'[64, 256, 64, 256, 5]',0,1,1.19421,6.64386,0.002083,0.002636,'None','model_Major3: output:-1 generation:0 index:2 count:0','adam','tanh'),(6,'20231030130533',317,128,-1,'[512, 8, 256, 64, 8, 5]',0,1,1.30347,6.40893,0.002687,0.003211,'None','model_Major3: output:-1 generation:0 index:3 count:0','adam','tanh'),(7,'20231030130722',360,32,-1,'[64, 512, 8, 8, 256, 8, 5]',0,1,1.33667,6.8708,0.002965,0.003321,'None','model_Major3: output:-1 generation:0 index:4 count:0','adam','tanh'),(8,'20231030131139',550,256,-1,'[256, 256, 32, 32, 5]',0,1,1.16031,5.52386,0.002095,0.002553,'None','model_Major3: output:-1 generation:0 index:5 count:0','adam','tanh'),(9,'20231030131414',271,32,-1,'[32, 8, 64, 5]',0,1,1.43839,8.26262,0.003199,0.003815,'None','model_Major3: output:-1 generation:0 index:6 count:0','adam','tanh'),(10,'20231030131619',472,256,-1,'[512, 32, 512, 32, 512, 32, 5]',0,1,1.14704,5.84072,0.002058,0.002535,'None','model_Major3: output:-1 generation:0 index:7 count:0','adam','tanh'),(11,'20231030131940',341,32,-1,'[256, 64, 512, 64, 5]',0,1,1.27742,5.70306,0.002562,0.002994,'None','model_Major3: output:-1 generation:0 index:8 count:0','adam','tanh'),(12,'20231030132322',291,128,-1,'[32, 32, 512, 512, 256, 512, 5]',0,1,1.2984,6.65199,0.002869,0.003169,'None','model_Major3: output:-1 generation:0 index:9 count:0','adam','tanh'),(13,'20231030132728',490,256,-1,'[256, 512, 32, 64, 5]',0,1,1.18554,5.42669,0.002109,0.002667,'None','model_Major3: output:-1 generation:0 index:10 count:0','adam','tanh'),(14,'20231030133019',431,256,-1,'[32, 256, 64, 256, 32, 5]',0,1,1.18384,5.53144,0.002258,0.002614,'None','model_Major3: output:-1 generation:0 index:11 count:0','adam','tanh'),(15,'20231030133238',336,32,-1,'[8, 256, 64, 64, 5]',0,1,1.21882,5.74539,0.002398,0.002782,'None','model_Major3: output:-1 generation:0 index:12 count:0','adam','tanh'),(16,'20231030133559',182,256,-1,'[256, 256, 5]',0,1,1.4549,8.92124,0.003427,0.00399,'None','model_Major3: output:-1 generation:0 index:13 count:0','adam','tanh'),(17,'20231030133647',369,128,-1,'[64, 8, 32, 512, 64, 5]',0,1,1.22397,5.72607,0.0025,0.002791,'None','model_Major3: output:-1 generation:0 index:14 count:0','adam','tanh'),(18,'20231030133902',213,32,-1,'[256, 512, 32, 32, 64, 32, 5]',0,1,1.20277,5.81768,0.002211,0.002871,'None','model_Major3: output:-1 generation:0 index:15 count:0','adam','tanh'),(19,'20231030134154',517,256,-1,'[64, 512, 32, 256, 8, 5]',0,1,1.29405,6.19127,0.002529,0.003133,'None','model_Major3: output:-1 generation:0 index:16 count:0','adam','tanh'),(20,'20231030134442',298,128,-1,'[256, 32, 8, 5]',0,1,1.38016,7.49372,0.002983,0.003569,'None','model_Major3: output:-1 generation:0 index:17 count:0','adam','tanh'),(21,'20231030134601',417,128,-1,'[256, 8, 256, 64, 5]',0,1,1.27296,6.00107,0.002545,0.002997,'None','model_Major3: output:-1 generation:0 index:18 count:0','adam','tanh'),(22,'20231030134827',242,256,-1,'[32, 64, 512, 256, 256, 8, 5]',0,1,1.2685,6.97861,0.002655,0.003158,'None','model_Major3: output:-1 generation:0 index:19 count:0','adam','tanh'),(23,'20231030135026',402,256,-1,'[256, 512, 32, 64, 5]',0,1,1.22733,5.81305,0.002207,0.002812,'None','model_Major3: output:-1 generation:1 index:0 count:0','adam','tanh'),(24,'20231030135302',540,256,-1,'[256, 256, 32, 32, 5]',0,1,1.16008,5.13724,0.002077,0.002519,'None','model_Major3: output:-1 generation:1 index:1 count:0','adam','tanh'),(25,'20231030135553',429,256,-1,'[32, 256, 64, 256, 32, 5]',0,1,1.26298,5.37725,0.002409,0.002959,'None','model_Major3: output:-1 generation:1 index:2 count:0','adam','tanh'),(26,'20231030135838',445,256,-1,'[256, 32, 32, 5]',0,1,1.39383,6.38223,0.002777,0.003538,'None','model_Major3: output:-1 generation:1 index:3 count:0','adam','tanh'),(27,'20231030140102',310,256,-1,'[256, 64, 256, 32, 5]',0,1,1.31003,5.43186,0.00264,0.003246,'None','model_Major3: output:-1 generation:1 index:4 count:0','adam','tanh'),(28,'20231030140309',337,256,-1,'[32, 512, 512, 32, 64, 5]',0,1,1.20004,5.84435,0.00222,0.002704,'None','model_Major3: output:-1 generation:1 index:5 count:0','adam','tanh'),(29,'20231030140753',325,256,-1,'[256, 32, 32, 5]',0,1,1.43684,8.8529,0.003086,0.003796,'None','model_Major3: output:-1 generation:1 index:6 count:0','adam','tanh'),(30,'20231030140929',461,256,-1,'[256, 64, 256, 32, 5]',0,1,1.29165,5.86662,0.002495,0.003064,'None','model_Major3: output:-1 generation:1 index:7 count:0','adam','tanh'),(31,'20231030141228',645,256,-1,'[32, 8, 512, 32, 64, 5]',0,1,1.18792,5.54445,0.002111,0.002624,'None','model_Major3: output:-1 generation:1 index:8 count:0','adam','tanh'),(32,'20231030141634',586,256,-1,'[256, 32, 32, 32, 5]',0,1,1.25541,5.73063,0.002264,0.002929,'None','model_Major3: output:-1 generation:1 index:9 count:0','adam','tanh'),(33,'20231030141955',457,256,-1,'[256, 256, 64, 256, 32, 5]',0,1,1.16084,5.50031,0.001942,0.002566,'None','model_Major3: output:-1 generation:1 index:10 count:0','adam','tanh'),(34,'20231030142422',889,256,-1,'[32, 8, 512, 32, 64, 5]',0,1,1.1361,5.10891,0.001888,0.002401,'None','model_Major3: output:-1 generation:1 index:11 count:0','adam','tanh'),(35,'20231030142923',423,256,-1,'[256, 32, 32, 5]',0,1,1.41775,7.45695,0.002937,0.003681,'None','model_Major3: output:-1 generation:1 index:12 count:0','adam','tanh'),(36,'20231030143126',463,256,-1,'[256, 64, 64, 256, 32, 5]',0,1,1.20355,5.52993,0.002253,0.002724,'None','model_Major3: output:-1 generation:1 index:13 count:0','adam','tanh'),(37,'20231030143408',545,256,-1,'[32, 512, 32, 64, 5]',0,1,1.20187,5.36279,0.002178,0.002672,'None','model_Major3: output:-1 generation:1 index:14 count:0','adam','tanh'),(38,'20231030143708',560,256,-1,'[256, 256, 32, 32, 5]',0,1,1.22121,5.60892,0.0022,0.002779,'None','model_Major3: output:-1 generation:1 index:15 count:0','adam','tanh'),(39,'20231030143949',381,256,-1,'[256, 256, 64, 256, 32, 5]',0,1,1.18224,5.61446,0.002148,0.002683,'None','model_Major3: output:-1 generation:1 index:16 count:0','adam','tanh'),(40,'20231030144211',220,256,-1,'[32, 512, 512, 32, 64, 5]',0,1,1.28063,6.24786,0.002512,0.003075,'None','model_Major3: output:-1 generation:1 index:17 count:0','adam','tanh'),(41,'20231030144425',417,256,-1,'[256, 32, 32, 5]',0,1,1.40244,6.97218,0.002892,0.003599,'None','model_Major3: output:-1 generation:1 index:18 count:0','adam','tanh'),(42,'20231030144609',379,256,-1,'[256, 512, 64, 256, 32, 5]',0,1,1.1815,5.4012,0.002019,0.002687,'None','model_Major3: output:-1 generation:1 index:19 count:0','adam','tanh'),(43,'20231030144853',566,256,-1,'[256, 512, 32, 64, 5]',0,1,1.14105,5.21962,0.002017,0.002468,'None','model_Major3: output:-1 generation:1 index:0 count:0','adam','tanh'),(44,'20231030145233',623,256,-1,'[256, 256, 32, 32, 5]',0,1,1.16303,5.75501,0.001897,0.002558,'None','model_Major3: output:-1 generation:1 index:1 count:0','adam','tanh'),(45,'20231030145541',538,256,-1,'[32, 256, 64, 256, 32, 5]',0,1,1.18312,5.2402,0.001909,0.002655,'None','model_Major3: output:-1 generation:1 index:2 count:0','adam','tanh'),(46,'20231030145845',330,256,-1,'[256, 32, 32, 5]',0,1,1.43126,8.07129,0.003045,0.003764,'None','model_Major3: output:-1 generation:1 index:3 count:0','adam','tanh'),(47,'20231030150008',288,256,-1,'[256, 64, 256, 32, 5]',0,1,1.32738,5.92244,0.002651,0.003268,'None','model_Major3: output:-1 generation:1 index:4 count:0','adam','tanh'),(48,'20231030150205',287,256,-1,'[32, 512, 512, 32, 64, 5]',0,1,1.24645,6.64907,0.002356,0.002918,'None','model_Major3: output:-1 generation:1 index:5 count:0','adam','tanh'),(49,'20231030150641',381,256,-1,'[256, 32, 32, 5]',0,1,1.41763,7.79363,0.002986,0.003687,'None','model_Major3: output:-1 generation:1 index:6 count:0','adam','tanh'),(50,'20231030150828',604,256,-1,'[256, 64, 256, 32, 5]',0,1,1.2176,5.59645,0.002281,0.002741,'None','model_Major3: output:-1 generation:1 index:7 count:0','adam','tanh'),(51,'20231030151145',755,256,-1,'[32, 8, 512, 32, 64, 5]',0,1,1.13685,5.15387,0.001941,0.002422,'None','model_Major3: output:-1 generation:1 index:8 count:0','adam','tanh'),(52,'20231030151621',210,256,-1,'[256, 32, 32, 32, 5]',0,1,1.43491,8.56189,0.003149,0.003866,'None','model_Major3: output:-1 generation:1 index:9 count:0','adam','tanh'),(53,'20231030151730',407,256,-1,'[256, 256, 64, 256, 32, 5]',0,1,1.20306,5.32165,0.002217,0.002703,'None','model_Major3: output:-1 generation:1 index:10 count:0','adam','tanh'),(54,'20231030152019',734,256,-1,'[32, 8, 512, 32, 64, 5]',0,1,1.1779,5.1723,0.002033,0.002595,'None','model_Major3: output:-1 generation:1 index:11 count:0','adam','tanh'),(55,'20231030152437',1000,256,-1,'[256, 32, 32, 5]',0,1,1.27891,6.03879,0.002253,0.003027,'None','model_Major3: output:-1 generation:1 index:12 count:0','adam','tanh'),(56,'20231030153004',394,256,-1,'[256, 64, 64, 256, 32, 5]',0,1,1.24439,5.30452,0.002447,0.002892,'None','model_Major3: output:-1 generation:1 index:13 count:0','adam','tanh'),(57,'20231030153234',728,256,-1,'[32, 512, 32, 64, 5]',0,1,1.16466,4.93625,0.001995,0.00251,'None','model_Major3: output:-1 generation:1 index:14 count:0','adam','tanh'),(58,'20231030153647',585,256,-1,'[256, 256, 32, 32, 5]',0,1,1.15251,5.20967,0.002017,0.002503,'None','model_Major3: output:-1 generation:1 index:15 count:0','adam','tanh'),(59,'20231030154011',445,256,-1,'[256, 256, 64, 256, 32, 5]',0,1,1.19722,5.76864,0.002054,0.002742,'None','model_Major3: output:-1 generation:1 index:16 count:0','adam','tanh'),(60,'20231030154338',226,256,-1,'[32, 512, 512, 32, 64, 5]',0,1,1.29217,5.91364,0.002621,0.003147,'None','model_Major3: output:-1 generation:1 index:17 count:0','adam','tanh'),(61,'20231030154601',446,256,-1,'[256, 32, 32, 5]',0,1,1.3956,6.50072,0.00284,0.003587,'None','model_Major3: output:-1 generation:1 index:18 count:0','adam','tanh'),(62,'20231030154805',316,256,-1,'[256, 512, 64, 256, 32, 5]',0,1,1.19624,5.87114,0.002185,0.002708,'None','model_Major3: output:-1 generation:1 index:19 count:0','adam','tanh');
/*!40000 ALTER TABLE `experiments` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2023-10-31  8:09:17
