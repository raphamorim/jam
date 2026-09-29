.PHONY: run install clean

run:
	bundle exec jekyll serve --livereload

install:
	bundle config set --local path vendor/bundle
	bundle install

clean:
	rm -rf _site
