@echo off

echo Compiling the crawler
javac -cp ".\lib\jsoup-1.11.3.jar" .\src\edu\ucr\cs235\crawler\*.java

echo Running the crawler
java -cp ".\src;.\lib\jsoup-1.11.3.jar" edu.ucr.cs235.crawler.PokemonApplication