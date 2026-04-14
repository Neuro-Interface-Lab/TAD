@ECHO OFF
REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set BUILDDIR=_build
set SOURCEDIR=.

if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't know where Sphinx is installed, try using the findstr
	echo.command to search for a file named sphinx-build.py or sphinx-build.exe
	echo.If you find it, uncomment the code below and start over.
	echo.
	REM set SPHINXBUILD=c:\path\to\sphinx\Scripts\sphinx-build.exe
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:end
popd
