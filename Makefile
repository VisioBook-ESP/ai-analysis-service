.PHONY: test test-unit test-integration coverage html clean help

# Commande par défaut
help:
	@echo "Commandes disponibles :"
	@echo "  make test           - Lancer tous les tests avec coverage"
	@echo "  make test-unit      - Lancer uniquement les tests unitaires"
	@echo "  make test-integration - Lancer uniquement les tests d'int�gration"
	@echo "  make html           - Ouvrir le rapport HTML de coverage"
	@echo "  make coverage       - Afficher le rapport de coverage d�taill�"
	@echo "  make clean          - Nettoyer les fichiers de test/cache"

# Lancer tous les tests (commande principale)
test:
	pytest

# Tests unitaires seulement (rapide)
test-unit:
	pytest -m unit

# Afficher le rapport de coverage dans le terminal
coverage:
	coverage report -m

# Ouvrir le rapport HTML
html:
	xdg-open htmlcov/index.html 2>/dev/null || firefox htmlcov/index.html 2>/dev/null || echo "Lancez d'abord: make test"

# Nettoyer les fichiers générés
clean:
	rm -rf htmlcov/
	rm -f .coverage
	rm -f coverage.xml
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
